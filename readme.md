# Multimodal Sentiment Analysis

本项目为多模态情感分析实验，旨在结合文本与图像数据，完成情感分析三分类任务。通过融合多模态信息，验证深度学习模型在情感分类任务中的性能提升。

## 项目链接

GitHub 仓库：[Multimodal Sentiment Analysis](https://github.com/Gnociew/Multimodal_sentiment_analysis)



## 数据集描述

- **数据组成**：
  - 数据文件夹包含训练文本和图像，每个文件以唯一的 `guid` 命名。
  - `train.txt`：训练数据的 `guid` 和对应的情感标签。
  - `test_without_label.txt`：测试数据的 `guid`，无标签。
- **数据规模**：
  - 4000条训练数据，511条测试数据。


## 仓库结构

```bash
Multimodal_Sentiment_Analysis/
├── Data.zip                       # 实验数据压缩包
├── code/                          # 实验代码
│   ├── MultiModal.ipynb           # 完整实验过程
│   ├── img_module.py              # 图像特征处理模块
│   ├── text_module.py             # 文本特征处理模块
│   ├── multiModal.py              # 多模态特征融合模块
│   ├── train_and_test.py          # 数据处理与模型训练模块
│   └── main.py                    # 主程序入口
├── requirements.txt               # 项目依赖
├── README.md                      # 项目说明文件
├── 实验报告.pdf                    
└── LICENSE                       
```


## 运行指南

### 环境要求
该实现基于 Python 3.11，运行代码需要以下依赖项：
- nltk==3.9.1
- pandas==2.2.3
- Pillow==11.1.0
- scikit_learn==1.6.1
- torch==2.5.1
- torchvision==0.20.1
- tqdm==4.67.1
- transformers==4.31.0

可以通过以下命令一键安装依赖：
```bash
pip install -r requirements.txt
```

### 数据准备

将数据集解压至代码运行目录下的 `Data/` 目录，确保以下结构：
```
Data/
├── data/
├── train.txt
└── test_without_label.txt
```

### 训练模型

- 运行以下命令进行模型训练：
    ```bash
    python main.py --model pretrain --fusion dual_cross_attention --seed 622
    ```
    其中
    - model 可以选择 ['custom', 'pretrained']
    - fusion 可以选择 ['concat', 'cross_attention', 'dual_cross_attention']
    - seed 可以输入任意 int 型整数
- 也可以进入 `MultiModal.ipynb` 中运行每个单元格 

---

## 实验结果

| 模型类型 | 融合方法            | 验证准确率 |
|----------|---------------------|------------|
| self     | 直接拼接            | 58.50%     |
| self     | 交叉注意力          | 62.00%     |
| self     | 双向交叉注意力      | 63.88%     |
| pretrain | 直接拼接            | 67.75%     |
| pretrain | 交叉注意力          | 70.25%     |
| pretrain | 双向交叉注意力      | 73.03%     |


如需更多详细信息，请参阅实验报告或联系项目维护者。