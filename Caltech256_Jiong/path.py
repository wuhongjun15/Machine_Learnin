import os
import sys
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 图片路径
TRAIN_IMAGE_PATH = os.path.join(sys.path[0], 'data', 'input', 'train_data')  # 训练数据路径
TEST_IMAGE_PATH = os.path.join(sys.path[0], 'data', 'input', 'test_data')  # 测试训练路径

# 部分参数
Torch_MODEL_NAME = "model.pkl"  # 保存的模型的名字
BATCH = 128  # 表示每批次有多少个数据
EPOCHS = 50  # 表示几轮次
LEARNING_RATE = 0.001
