from text_module import MultiHeadAttention

import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossModalAttention, self).__init__()

        self.mha = MultiHeadAttention(d_model, nhead, dropout=dropout)
  
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, txt_feats, mask=None):
        # 图像特征做 query，文本特征做 key/value
        out = self.mha(query=img_feats, key=txt_feats, value=txt_feats, mask=mask)

        # 残差 + LayerNorm
        out = self.norm(img_feats + self.dropout(out))
        return out
    

class DualCrossModalAttention(nn.Module):
    def __init__(self, MultiHeadAttnClass=None, d_model=512, nhead=8, dropout=0.1):
        super(DualCrossModalAttention, self).__init__()
        
        # 图像 -> 查询文本
        self.cross_attn_img2txt = MultiHeadAttnClass(d_model, nhead, dropout=dropout)
        # 文本 -> 查询图像
        self.cross_attn_txt2img = MultiHeadAttnClass(d_model, nhead, dropout=dropout)
        
        # 层归一化
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_txt = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, txt_feats, mask_txt=None):
        img2txt_out = self.cross_attn_img2txt(query=img_feats, key=txt_feats, value=txt_feats, mask=mask_txt)
        out_img = self.norm_img(img_feats + self.dropout(img2txt_out))

        txt2img_out = self.cross_attn_txt2img(query=txt_feats, key=out_img, value=out_img,)
        out_txt = self.norm_txt(txt_feats + self.dropout(txt2img_out))

        return out_img, out_txt
    
class MultiModalModel_self(nn.Module):
    def __init__(self,
                 image_model: nn.Module,
                 text_model: nn.Module,
                 fusion = 'dual_cross_attention',
                 d_model=512,
                 nhead=8,
                 dropout=0.1,
                 num_classes=3):
        super(MultiModalModel_self, self).__init__()

        self.image_model = image_model     
        self.text_model = text_model      
        self.cross_attn = CrossModalAttention(d_model, nhead, dropout)
        self.dual_cross_attn = DualCrossModalAttention(MultiHeadAttnClass=MultiHeadAttention, d_model=d_model, nhead=nhead, dropout=dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.classifier = nn.Linear(d_model*2, num_classes)
        self.fusion_method = fusion

    def forward(self, images, text_ids, mask):
        # 图像特征 => [B, 1, d_model]
        img_feats = self.image_model(images)
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1)
    
        # 文本特征 => [B, txt_len, d_model]
        txt_feats = self.text_model(text_ids, src_mask=mask)
    
        #### 特征拼接 ####

        ## 1)直接拼接
        if self.fusion_method == 'concat':
            fused_feats = torch.cat([img_feats, txt_feats], dim=1)  # [B, 1 + txt_len, d_model]
            fused_feats = fused_feats.mean(dim=1)  # 平均池化
            logits = self.fc(fused_feats)


        # 2)交叉注意力拼接
        if self.fusion_method == 'cross_attention':
            fused_img_feats = self.cross_attn(img_feats, txt_feats, mask)  # [B, 1, d_model]
            fused_img_feats = fused_img_feats.squeeze(1)
            logits = self.fc(fused_img_feats)

        # 3)双向交叉注意力拼接
        # out_img => [B, img_len, d_model]
        # out_txt => [B, txt_len, d_model]
        if self.fusion_method == 'dual_cross_attention':
            out_img, out_txt = self.dual_cross_attn(img_feats, txt_feats, mask_txt=mask)

            fused_img = out_img.mean(dim=1)
            fused_txt = out_txt.mean(dim=1) 

            fused = torch.cat([fused_img, fused_txt], dim=-1)  
            logits = self.classifier(fused)       
        
        return logits
    
class MultiModalModel_pretrain(nn.Module):
    def __init__(self, image_model, text_model,fusion='dual_cross_attention', d_model=512,nhead = 8, dropout=0.1, num_classes=3):
        super(MultiModalModel_pretrain, self).__init__()
        self.image_model = image_model
        self.text_model = text_model

        self.cross_attn = CrossModalAttention(d_model, nhead, dropout)
        self.dual_cross_attn = DualCrossModalAttention(MultiHeadAttnClass=MultiHeadAttention, d_model=d_model, nhead=nhead, dropout=dropout)

        self.classifier = nn.Linear(d_model*2, num_classes)

        self.fusion = nn.Linear(2*d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.fc = nn.Linear(d_model, num_classes)

        self.fusion_method = fusion

    def forward(self, images, input_ids, mask_txt=None):
        # 图像特征 => [B, img_len, d_model]
        img_feats = self.image_model(images)
        img_feats = img_feats.unsqueeze(1)

        # 文本特征 => [B, txt_len, d_model]
        txt_feats = self.text_model(input_ids)

        # 直接拼接
        if self.fusion_method == 'concat':
            fused_feats = torch.cat([img_feats, txt_feats], dim=1)  # [B, 1 + txt_len, d_model]
            fused_feats = fused_feats.mean(dim=1)  # 平均池化
            logits = self.fc(fused_feats)

        # # 交叉注意力
        if self.fusion_method == 'cross_attention':
            fused_img_feats = self.cross_attn(img_feats, txt_feats, mask_txt)
            fused_img_feats = fused_img_feats.squeeze(1)
            logits = self.fc(fused_img_feats)


        # 双向 Cross-Attention
        # print("img_feats.shape =", img_feats.shape)
        # print("txt_feats.shape =", txt_feats.shape)
        if self.fusion_method == 'dual_cross_attention':
            out_img, out_txt = self.dual_cross_attn(img_feats, txt_feats, mask_txt=mask_txt)
            fused_img = out_img[:,0,:] 
            fused_txt = out_txt[:,0,:]

            fused = torch.cat([fused_img, fused_txt], dim=-1) 
            logits = self.classifier(fused)                 
        
        return logits
