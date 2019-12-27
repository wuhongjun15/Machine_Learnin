import torch
import os
import random
import shutil

from utils import progress_bar
from path import MODEL_PATH, Torch_MODEL_NAME


# 从训练数据中遍历所有的类（子文件夹），从每个类里面随机选取一些数据（文件）拷贝出来作为测试数据
def creat_test_data(train_data_path, test_data_path):
    print("Test data creating!")
    # 检查训练数据是否存在
    if not os.path.isdir(train_data_path):
        print("sorry, there is not train data!")
        exit(2)
    # 生成测试数据文件夹
    if not os.path.isdir(test_data_path):
        os.makedirs(test_data_path)
    # 遍历训练数据下的所有类（子文件夹）
    folder_list = os.listdir(train_data_path)
    for folder in folder_list:
        target_folder = os.path.join(test_data_path, folder)
        # 在测试数据中生成对应的类
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        # 从每个类（子文件夹）中随机选4个数据作为测试数据拷贝到对应类（文件夹）中
        image_path = os.path.join(train_data_path, folder)
        image_list = os.listdir(image_path)
        images = random.sample(image_list, 4)
        for image in images:
            shutil.copy(os.path.join(image_path, image), target_folder)
    print("Test data created successfully!")


# 检查模型是否存在，是否覆盖
def check(path, overwrite=False):  # overwrite model or not
    if overwrite:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    if not os.path.exists(path):
        os.makedirs(path)


# 在需要时，将模型保存到本地
def save_model(network, path, name=Torch_MODEL_NAME, overwrite=False):
    print('Model Saving..')
    check(path, overwrite)
    torch.save(network, os.path.join(path, name))
    print('Model %s Saved..' % Torch_MODEL_NAME)


# 进行测试
def test(net, testloader, epochs, loss_fn, best_acc):
    cuda_is_avail = torch.cuda.is_available()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x_test, y_test) in enumerate(testloader):
            # 如果cuda可用，用另一种方式变换数据，为后面调用GPU处理数据加速服务
            if cuda_is_avail:
                x_test, y_test = x_test.to('cuda'), y_test.to('cuda')
            # 网络前向传递，计算输出
            outputs = net(x_test)
            # 计算输出与标签之间得损失
            loss = loss_fn(outputs, y_test)

            # 计算准确率
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_test.size(0)
            correct += predicted.eq(y_test).sum().item()

            accuracy = 100. * correct / total
            # 打印当前批次的准确率

            # print('test: epoch: %d ,batch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #       % (epochs, batch_idx, test_loss / (batch_idx + 1), accuracy, correct, total))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)(model is testing!)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save model
    acc = 100. * correct / total
    if acc > best_acc:
        save_model(net, MODEL_PATH, overwrite=True)
        best_acc = acc
    return best_acc
