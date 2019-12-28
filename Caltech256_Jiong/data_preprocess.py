# 数据预处理
# 将图片信息及标签处理成可训练模式
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


class DataPreprocess(object):
    def __init__(self, image_path, batch=1):
        self.path = image_path
        self.batch = batch

    # 输入图片路径，返回处理之后的dataLoader
    def image2data(self):
        # [-1,1]处理,参数来源于开发文档的示例
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 我看了大多数的代码都是resize成224*224
            transforms.ToTensor(),  # 读取图片像素并且转化为0-1的数字
            normalize,
        ])
        dataset = ImageFolder(self.path, transform=transform)

        # torch.utils.data.DataLoader类可以将list类型的输入数据封装成Tensor数据格式，以备模型使用。
        # 数据量太大，要分成批次进行处理
        # 创建DataLoader迭代器
        data_loader = DataLoader(dataset, batch_size=self.batch, shuffle=True, pin_memory=True)
        category_num = int(len(dataset.class_to_idx))
        return data_loader, category_num
