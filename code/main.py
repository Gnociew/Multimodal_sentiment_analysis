import argparse
from train_and_test import trian_module

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Choose model and hyperparameters for training")

    # 添加命令行参数
    parser.add_argument('--model', type=str, choices=['custom', 'pretrained'], required=True,
                        help='Choose the model: custom or pretrained')
    
    parser.add_argument('--fusion',type=str,choices=['concat', 'cross_attention', 'dual_cross_attention'],required=True, 
        help="Choose feature fusion method: 'concat' (direct concatenation), 'cross_attention', or 'dual_cross_attention'")
    
    parser.add_argument('--seed', type=int, default=622, 
                        help='Seed for the model')
    

    # 解析命令行参数
    args = parser.parse_args()
    print(f"Selected Model: {args.model}")
    print(f"Feature Fusion Method: {args.fusion}")
    print(f"Seed: {args.seed}")
  
    trian_module(args.model, args.fusion, args.seed)
    
    

if __name__ == "__main__":
    main()
