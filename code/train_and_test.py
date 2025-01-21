import os
from PIL import Image
import pandas as pd

import re

from nltk.corpus import stopwords
import string
from transformers import BertTokenizer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from sklearn.preprocessing import LabelEncoder
from torchvision.ops import sigmoid_focal_loss


from torchvision import transforms

from tqdm import tqdm

from img_module import ResNetBlock, ResNet, PretrainedViTExtractor
from text_module import Encoder,BertTextEncoder
from multiModel import MultiModalModel_self,MultiModalModel_pretrain


def load_data(data_dir, train_file):
    # 存储训练集数据和标签
    dataset = []
    
    # 读取 train.txt 文件
    train_path = os.path.join(data_dir, train_file)
    with open(train_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            guide, tag = line.strip().split(',')
            
            # 构造文件路径
            img_path = os.path.join(data_dir, "data", f"{guide}.jpg")
            txt_path = os.path.join(data_dir, "data",f"{guide}.txt")
            
            # 加载图片和文本数据
            if os.path.exists(img_path) and os.path.exists(txt_path):
                # 读取图片
                img = Image.open(img_path).convert('RGB')
                # 读取文本
                with open(txt_path, 'r' ,encoding='utf-8', errors='replace') as txt_file:
                    text = txt_file.read()
                # 添加到数据集
                dataset.append({
                    'guide': guide,
                    'image': img,
                    'text': text,
                    'tag': tag
                })
    return dataset

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # 去除URL
    text = re.sub(r'@\w+', '', text)    # 去除@用户
    text = re.sub(r'\bRT\b', '', text)  # 去除RT转发标记
    text = re.sub(r'[^\w\s]', '', text) # 去除特殊字符和表情符号
    text = text.strip()                 # 去除多余空格
    return text

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./bert_model/')
stop_words = set(stopwords.words('english'))    # 英语停用词列表
punctuation_table = str.maketrans('', '', string.punctuation)   # 翻译表，将所有标点符号替换为空字符

def preprocess_text_for_w2v(text):
    clean_text(text)
    if not isinstance(text, str):
        return []
    text = text.lower().translate(punctuation_table)
    tokens = tokenizer.tokenize(text)  
    return [token for token in tokens if token.isalpha() and token not in stop_words] 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
])

class SummarizationDataset(Dataset):
    def __init__(self, df, image_col='image', text_col='text',target_col='tag', max_len=20, transform=None, tokenizer=None):
        self.df = df
        self.images = df[image_col].tolist()
        self.texts = df[text_col].tolist()
        self.labels = df[target_col].tolist()
        self.max_len = max_len
        
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        
        text_to_tokenize = ' '.join(self.texts[idx])
        encoded_inputs = tokenizer(
            text_to_tokenize,             
            max_length=20,             
            padding='max_length',       
            truncation=True,            
            return_tensors="pt"         
        )   

        return {
            'image': image,
            'input_ids': encoded_inputs['input_ids'].squeeze(0),
            'attention_mask': encoded_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }
    
data_dir = "Data"
train_file = "train.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./res_best_models"
os.makedirs(model_path, exist_ok=True)
graph_path = "./res_graphs"
os.makedirs(graph_path, exist_ok=True)

num_epochs = 30
batch_size = 32
learning_rate = 0.0001
patience = 3
dropout_prob = 0.8

vocab_size = tokenizer.vocab_size


def trian_module(module_name, fusion, seed):

    dataset = load_data(data_dir, train_file)
    df = pd.DataFrame(dataset)
    df['tokens'] = df['text'].apply(preprocess_text_for_w2v)
    label_encoder = LabelEncoder()
    df['tag_encoded'] = label_encoder.fit_transform(df['tag'])
    dataset = SummarizationDataset(df, image_col='image', text_col='tokens', target_col='tag_encoded', transform=transform, tokenizer=tokenizer)

    train_size = int(0.8 * len(dataset))  # 80% 用于训练
    val_size = len(dataset) - train_size   # 20% 用于验证
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if module_name == 'custom':
        resnet_feature_extractor = ResNet(
            block=ResNetBlock, 
            layers=[2, 2, 2, 2], 
            feature_dim=512, 
            dropout_prob=0.5
        )

        # 2. 定义文本模型
        text_feature_extractor = Encoder(
            vocab_size=vocab_size,     # 根据数据集大小来
            d_model=512, 
            nhead=8, 
            num_layers=2, 
            dim_feedforward=2048, 
            dropout=0.1, 
            max_len=512
        )

        model = MultiModalModel_self(
            image_model=resnet_feature_extractor,
            text_model=text_feature_extractor,
            fusion = fusion,
            d_model=512,
            nhead=8,
            dropout=0.1
        ).to(device)
    else:
        resnet_feature_extractor = PretrainedViTExtractor()

        text_feature_extractor = BertTextEncoder(
            feature_dim=512,
            freeze_bert=True
        )

        model = MultiModalModel_pretrain(
            image_model=resnet_feature_extractor,
            text_model=text_feature_extractor,
            fusion = fusion,
            d_model=512,
            nhead = 8, 
            dropout=0.1
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    # 早停参数
    best_val_acc = 0
    epochs_no_improve = 0  
    early_stop = False  


    for epoch in range(num_epochs):
        # 训练模式
        model.train()

        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'轮次 {epoch+1}/{num_epochs}'):
            image = batch['image'].to(device)
            text = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].long().to(device)

            optimizer.zero_grad()   # 清空梯度

            outputs = model(image, text, attention_mask)
            labels_one_hot = F.one_hot(labels, num_classes=3).float()
            loss = sigmoid_focal_loss(
                outputs,  # 模型的输出 (logits)，未经过 sigmoid
                labels_one_hot,  # 标签 (0 或 1)，与 inputs 的 shape 相同
                alpha=0.25,  # 平衡因子
                gamma=2.0,  # 调节因子
                reduction="mean"  # 'none', 'mean', or 'sum'
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = round(total_train_loss / len(train_loader), 4)
        train_losses.append(avg_train_loss)
        print(f'轮次 {epoch+1}, 训练损失: {avg_train_loss}')


        # 验证模式
        model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():   # 不需要更新参数，禁用梯度计算
            for batch in tqdm(val_loader, desc=f'轮次 {epoch+1}/{num_epochs}'):
                image = batch['image'].to(device)
                text = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].long().to(device)
                
                outputs = model(image,text, attention_mask)
                labels_one_hot = F.one_hot(labels, num_classes=3).float()
                loss = sigmoid_focal_loss(
                    outputs,  # 模型的输出 (logits)，未经过 sigmoid
                    labels_one_hot,  # 标签 (0 或 1)，与 inputs 的 shape 相同
                    alpha=0.25,  # 平衡因子
                    gamma=2.0,  # 调节因子
                    reduction="mean"  # 'none', 'mean', or 'sum'
                )
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)   # 提取预测的类别索引
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = round(val_loss / len(val_loader), 4)
        val_losses.append(avg_val_loss)
        val_acc = round(100 * correct / total, 2)
        print(f'验证损失: {avg_val_loss}, 准确率: {val_acc}%')

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            file_path = os.path.join(model_path, f'best_model.pth')
            torch.save(model.state_dict(), file_path)
            print('验证准确率升高，保存当前模型')
        else:
            epochs_no_improve += 1
            print(f'验证准确率未提升，未提升的 epoch 数：{epochs_no_improve}')

        if epochs_no_improve >= patience:
            print('早停触发')
            early_stop = True
            break

    if early_stop:
        print(f'由于早停机制，提前结束训练，在第 {epoch+1} 个 epoch 停止。')
    else:
        print('训练完成')
