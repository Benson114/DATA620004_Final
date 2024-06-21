# DATA620004-Final-Task2

## Instruction

本项目为复旦大学研究生课程DATA620004——神经网络和深度学习期末作业Task2的代码仓库

## Requirements

```shell
pip install -r requirements.txt
```

## How to Run

### 模型训练

```shell
python main.py -m [模型类型] -c [是否清除tensorboard记录]
# 例
python main.py -m SimpleCNN -c True
python main.py -m ViT -c False
```

### 模型测试

请确保已有训练完成自动保存的模型权重

```shell
python test.py -m [模型类型]
```
