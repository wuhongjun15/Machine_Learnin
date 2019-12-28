import torch
import torch.nn as nn
import sys
import os
import torchvision
import argparse
import torch.optim as optim
from torch.optim import Adam

from data_preprocess import DataPreprocess
from Net import myModel
from train import train
from test import test, creat_test_data
from path import MODEL_PATH, Torch_MODEL_NAME, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, BATCH, EPOCHS, LEARNING_RATE


# 判断CUDA是否可用
cuda_is_avail = torch.cuda.is_available()
# 是调用较流行的CNN网络，还是使用自己搭建的CNN网络
use_cnn = True

# 运行时，命令行配置超参,不设置则按默认值运行
parser = argparse.ArgumentParser(description='PyTorch Caltech256 Training')
parser.add_argument("-e", "--EPOCHS", default=EPOCHS, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=BATCH, type=int, help="batch size")
parser.add_argument('-lr', "--LEARNING_RATE", default=LEARNING_RATE, type=float, help='learning rate')
parser.add_argument('-lm', "--LOAD_MODEL", default=0, type=int, help='if or not load last best model')
args = parser.parse_args()

# 生成测试数据
creat_test_data(TRAIN_IMAGE_PATH, TEST_IMAGE_PATH)

# 将图像处理成可用数据
# Category_num表示训练数据集中的类别个数，对于caltech256来说，应该有257个类别
train_data, category_num = DataPreprocess(TRAIN_IMAGE_PATH, args.BATCH).image2data()
test_data, _ = DataPreprocess(TEST_IMAGE_PATH, args.BATCH).image2data()

# 是否加载之前保存模型模型
if args.LOAD_MODEL and os.path.isfile(os.path.join(MODEL_PATH, Torch_MODEL_NAME)):
    print("Loading model: "+Torch_MODEL_NAME)
    cnn = torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
    print("model load successful!")
else:
    # 定义DenseNet实例,加载预训练模型
    if use_cnn:
        cnn = torchvision.models.densenet121(pretrained=True)
        for param in cnn.parameters():
            param.requires_grad = False
        num_features = cnn.classifier.in_features
        # 更改最后一层
        cnn.classifier = nn.Linear(num_features, category_num)
    else:
        cnn = myModel()  # 暂时还没写更好的

if cuda_is_avail:
    # cnn.to('cuda')
    cnn.cuda()

# 优化器
optimizer = Adam(cnn.parameters(), lr=args.LEARNING_RATE, betas=(0.9, 0.999))  # 选用AdamOptimizer
# optimizer = optim.SGD(cnn.parameters(), lr=args.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
best_acc = 0  # best test accuracy
for i in range(args.EPOCHS):
    cnn = train(cnn, train_data, i, optimizer, loss_fn)
    best_acc = test(cnn, test_data, i, loss_fn, best_acc)

