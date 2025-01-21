import torch.nn as nn
from transformers import ViTModel

# 残差块
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_prob=0.5, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()     # 调用父类方法

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),

            nn.Dropout(dropout_prob),
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.residual(x) 
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out) 
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, feature_dim=512, dropout_prob=0.5):
        # layers: 列表，每个block的数量
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(block, 64, layers[0], dropout_prob)
        self.layer2 = self.make_layer(block, 128, layers[1], dropout_prob=dropout_prob, stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], dropout_prob=dropout_prob, stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], dropout_prob=dropout_prob, stride=2)

        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512, feature_dim)
            # nn.Linear(512, num_classes)
        )


    def make_layer(self, block, out_channels, num_blocks, dropout_prob=0.5, stride=1):
        # 下采样
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        # 构建层，每个层包含多个残差块
        layers = []

        # 第一个残差块，（也许）需要下采样
        layers.append(block(self.in_channels, out_channels, dropout_prob, stride, downsample))
        self.in_channels = out_channels
        # 后num_blocks-1个残差块
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, dropout_prob ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_layers(x)
        return x

class PretrainedViTExtractor(nn.Module):
    def __init__(self, feature_dim=512, pretrained_path='./vit_model'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_path)
        self.fc = nn.Linear(768, feature_dim) 

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        feats = self.fc(pooled)
        return feats