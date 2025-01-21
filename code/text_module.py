
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引数组：[max_len, 1]

        # 分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 生成正余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数使用cos
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # 将pe注册为模型的缓冲区，在训练和推理时不会作为参数更新，但可以通过模型保存和加载。
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1) # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :seq_len, :] # 广播
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度

        # Q, K, V 的投影层，用于将 query, key, value 投影到高维空间
        # [batch_size, seq_len, d_model]
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 将多头输出合并后再投影回原始维度
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
  
        B, Q_len, _ = query.size()
        B, K_len, _ = key.size()
        B, V_len, _ = value.size()

        # 线性变换
        # [batch_size, seq_len, d_model] 
        Q = self.w_q(query) 
        K = self.w_k(key)    
        V = self.w_v(value)  

        # 拆分多头
        Q = Q.view(B, Q_len, self.nhead, self.d_k).transpose(1, 2)  # [B, nhead, Q_len, d_k]
        K = K.view(B, K_len, self.nhead, self.d_k).transpose(1, 2)  # [B, nhead, K_len, d_k]
        V = V.view(B, V_len, self.nhead, self.d_k).transpose(1, 2)  # [B, nhead, V_len, d_k]

        # 计算 Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用 mask
        # 保证 pad 或 decoder 中不可见的 token 不参与 attention
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)   # [B, K_len] -> [B, 1, 1, K_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)                # [B, Q_len, K_len] -> [B, 1, Q_len, K_len]
            
            scores = scores.masked_fill(mask == 0, float('-inf'))   # 将 mask 为 0 的位置填充为负无穷大 ，在 softmax 中变为 0

        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.matmul(attn, V)  # [B, nhead, Q_len, d_k]

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, Q_len, self.d_model)

        # 输出
        # 原始维度[batch_size, Q_len, d_model]
        out = self.fc_out(out)
        return out
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) # 增加网络宽度以捕获更复杂的特征模式
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 注意力机制
        attn_out = self.self_attn(src, src, src, mask=src_mask)
        out1 = self.norm1(src + self.dropout(attn_out))
        
        # 前馈网络
        out2 = self.linear1(out1)
        out = self.norm2(out1 + self.dropout(out2)) # 残差连接
        
        return out
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, max_len=512):
        super(Encoder, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [batch_size, src_len]

        x = self.embed_tokens(src)  # [batch_size, src_len, d_model]
        x = self.pos_emb(x) # 位置编码层，为嵌入向量引入位置信息
        x = self.dropout(x)

        # encoder 层
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x
    
class BertTextEncoder(nn.Module):
    def __init__(self, feature_dim=768, freeze_bert=False, pretrained_model_path='./bert_model1/'):
        super(BertTextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_path)
        
        # 对部分层 freeze
        if freeze_bert:
            for name, param in self.bert.named_parameters():
                if "encoder.layer.9." not in name and "encoder.layer.10." not in name and "encoder.layer.11." not in name:
                    param.requires_grad = False
        
        self.proj = nn.Linear(768, feature_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = outputs.last_hidden_state       # [B, seq_len, 768]
        last_hidden_state = self.proj(last_hidden_state)    # [B, seq_len, feature_dim]

        return last_hidden_state