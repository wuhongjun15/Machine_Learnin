import torch
from torch.autograd import Variable
from utils import progress_bar


def train(net, dataloader, epochs, optimizer, loss_fn):
    # 判断CUDA是否可用
    cuda_is_avail = torch.cuda.is_available()

    # 训练并评估模型
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # 使用enumerate访问可遍历的数组对象
    for batch_idx, (x_train, y_train) in enumerate(dataloader):
        # print('Training: Epoch: %d, batch: %d' % (epochs, batch_idx))
        # 如果cuda可用，将数据类型变换为Variable，为后面调用GPU加速训练服务
        if cuda_is_avail:
            # 将数据变为GPU可用的数据
            # x_train, y_train = x_train.to('cuda'), y_train.to('cuda')
            x_train, y_train = Variable(x_train.cuda()), Variable(y_train.cuda())
        x_train = x_train.float()

        # 网络的向前传播
        outputs = net(x_train)

        optimizer.zero_grad()  # 每个batch都要将网络中的所有梯度置为0，原因可百度
        # 将输出的outputs和原来导入的labels作为loss函数的输入,得到损失
        loss = loss_fn(outputs, y_train)
        # 回传得到的损失
        loss.backward()
        # 回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，optimizer.step()就是用来更新参数的。
        # optimizer.step()后，你就可以从optimizer.param_groups[0][‘params’]里面看到各个层的梯度和权值信息。
        optimizer.step()

        # 计算准确率
        train_loss += loss.item()
        _, prediction = outputs.max(1)
        total += y_train.size(0)
        correct += prediction.eq(y_train).sum().item()

        accuracy = 100. * correct / total
        # print('train: epoch: %d ,batch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #       % (epochs, batch_idx, train_loss / (batch_idx + 1), accuracy, correct, total))
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc:%.3f%% (%d/%d) (model is training!)'
                     % (train_loss/(batch_idx+1), accuracy, correct, total))
    return net

