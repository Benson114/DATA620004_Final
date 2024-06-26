# DATA620004-Final-Task1

## Instruction

本项目为复旦大学研究生课程DATA620004——神经网络和深度学习期末作业Task1的代码仓库

## Requirements

```shell
pip install -r requirements.txt
```

## How to Run

### 模型训练

```shell
python main.py -p [预训练类型]
# 例
python main.py -p None
python main.py -p SL
```

### 模型测试

请确保已有训练完成自动保存的模型权重

```shell
python test.py -p [预训练类型]
```
