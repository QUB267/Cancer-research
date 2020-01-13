# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import shutil
import os
import numpy as np
from sklearn .metrics import confusion_matrix
random_state = 42
np.random.seed(random_state)

# kaggle原始数据集地址
original_dataset_dir = 'E:\\train'
total_num = int(len(os.listdir(original_dataset_dir)) / 2)
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)

# 待处理的数据集地址
base_dir = 'E:\data2'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
#猫狗划分
sub_dirs = ['cats', 'dogs']
animals = ['cats', 'dogs']
cat_idx = random_idx[:int(total_num)]
dog_idx = random_idx[int(total_num):]
numbers = [cat_idx, dog_idx]
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for animal in animals:
        animal_dir = os.path.join(base_dir, animal)
        if not os.path.exists(animal_dir):
            os.mkdir(animal_dir)
        fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
        if (animal=='dogs'):
         random_state =1
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(animal_dir, fname)
            shutil.copyfile(src, dst)
# 训练集、测试集的划分
sub_dirs = ['train', 'test', 'validation ']
animals = ['cats', 'dogs']
train_idx = random_idx[:int(total_num * 0.9)]
test_idx = random_idx[int(total_num * 0.9)]
validation_idx = random_idx[int(total_num * 0.1)]
numbers = [train_idx, test_idx, validation_idx]
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for animal in animals:
        animal_dir = os.path.join(dir, animal)
        if not os.path.exists(animal_dir):
            os.mkdir(animal_dir)
        fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(animal_dir, fname)
            shutil.copyfile(src, dst)
    print(animal_dir + ' total images : %d' % (len(os.listdir(animal_dir))))
    # coding=utf-8

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

epochs = 10 # 训练次数
batch_size = 4  # 批处理大小
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
PATH='E:\python ese\c_d\model.pt'
# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='E:\data2\\train',
                                     transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

test_dataset = datasets.ImageFolder(root='E:\data2\\test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
if(os.path.exists('model.pt')):
    net=torch.load('model.pt')

if use_gpu:
    net = net.cuda()
print(net)

# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def train():

    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
        # 模型测试
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        net.eval()
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))

    torch.save(net, 'model.pt')

def reload_net():

    trainednet = torch.load('model.pt')

    return trainednet


def test():
    correct = 0
    test_loss = 0.0
    test_total = 0
    classnum=2
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    net.eval()
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)
        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        correct_num = correct*torch.ones((1, classnum))
        test_total_num = test_total*torch.ones((1, classnum))
        specificity = (correct_num-acc_num)/(test_total_num-target_num)
        print(recall)
        print(specificity)

    print(' loss: %.3f  acc: %.3f ' % (test_loss / test_total, 100 * correct / test_total))
    print('recall', " ".join('%s' % id for id in recall))
    print('precision', " ".join('%s' % id for id in precision))
    print('specificity', " ".join('%s' % id for id in specificity))
    print('F1', " ".join('%s' % id for id in F1))



test()











