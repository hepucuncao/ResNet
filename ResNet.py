# coding=gbk
# 1.���ر�Ҫ�Ŀ�
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import argparse

# 2.������
BATCH_SIZE = 32#ÿ����������� һ���Զ��ٸ�
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#ʹ��GPU
EPOCHS =4 #ѵ�����ݼ����ִ�
# 3.ͼ����
pipeline = transforms.Compose([transforms.ToTensor(), #��ͼƬת��ΪTensor
])

# 4.���أ���������
from torch.utils.data import DataLoader

# �������ݼ�
train_set = datasets.MNIST("MNIST",train=True,download=True,transform=pipeline)
test_set = datasets.MNIST("MNIST",train=False,download=True,transform=pipeline)

# �������ݼ� һ���Լ���BATCH_SIZE������˳�������
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)


# 5.��������ģ��
class ResBlk(nn.Module):  # ����Resnet Blockģ��
    def __init__(self, ch_in, ch_out, stride=1):  # ��������ǰ�ȵ�֪����������ʹ����������趨
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # ��ʼ��

        # we add stride support for resbok, which is distinct from tutorials.
        # ����resnet����ṹ����2����block����ṹ ��һ���� ����˴�С3*3,����Ϊ1����Ե��1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # ����һ�����������Ϣͨ��BatchNorm2d
        self.bn1 = nn.BatchNorm2d(ch_out)
        # �ڶ��������յ�һ������������һ��
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # ȷ������ά�ȵ������ά��
        self.extra = nn.Sequential()  # �Ƚ�һ���յ�extra
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):  # ����ֲ���ǰ��������
        out = F.relu(self.bn1(self.conv1(x)))  # �Ե�һ������������پ���relu����
        out = self.bn2(self.conv2(out))  # �ڶ���������������
        out = self.extra(x) + out  # ��x����extra����2�飨block���������ԭʼֵ�������
        out = F.relu(out)  # ����relu
        return out


class ResNet18(nn.Module):  # ����resnet18��

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # ���ȶ���һ�������
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # followed 4 blocks ����4��resnet����ṹ��������������2��
        self.blk1 = ResBlk(32, 64, stride=1)
        self.blk2 = ResBlk(64, 128, stride=1)
        self.blk3 = ResBlk(128, 256, stride=1)
        self.blk4 = ResBlk(256, 256, stride=1)
        self.outlayer = nn.Linear(256 * 1 * 1, 10)  # �����ȫ���Ӳ�

    def forward(self, x):  # ����������ǰ����

        x = F.relu(self.conv1(x))  # �Ⱦ�����һ����

        x = self.blk1(x)  # Ȼ��ͨ��4��resnet����ṹ
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)  # ƽ��һάֵ
        x = self.outlayer(x)  # ȫ���Ӳ�

        return x

# 6.�����Ż���
model = ResNet18().to(DEVICE)#����ģ�Ͳ���ģ�ͼ��ص�ָ���豸��

optimizer = optim.Adam(model.parameters(),lr=0.001)#�Ż�����

criterion = nn.CrossEntropyLoss()
# 7.ѵ��
def train_model(model,device,train_loader,optimizer,epoch):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()


    model.train()#ģ��ѵ��
    for batch_index,(data ,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)#����DEVICE��ȥ
        optimizer.zero_grad()#�ݶȳ�ʼ��Ϊ0
        output = model(data)#ѵ����Ľ��
        loss = criterion(output,target)#����������ʧ
        loss.backward()#���򴫲� �õ��������ݶ�ֵ
        optimizer.step()#�����Ż�
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))
            if args.dry_run:
                break
# 8.����
def test_model(model,device,text_loader):
    model.eval()#ģ����֤
    correct = 0.0#��ȷ��
    global Accuracy
    text_loss = 0.0
    with torch.no_grad():#��������ݶȣ�Ҳ������з��򴫲�
        for data,target in text_loader:
            data,target = data.to(device),target.to(device)#����device��
            output = model(data)#�����Ľ��
            text_loss += criterion(output,target).item()#���������ʧ
            pred = output.argmax(dim=1)#�ҵ����������±�
            correct += pred.eq(target.view_as(pred)).sum().item()#�ۼ���ȷ��ֵ
        text_loss /= len(test_loader.dataset)#��ʧ��/���ص����ݼ�������
        Accuracy = 100.0*correct / len(text_loader.dataset)
        print("Test__Average loss: {:4f},Accuracy: {:.3f}\n".format(text_loss,Accuracy))
# 9.����

for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)

# 10.������ò���
torch.save(model.state_dict(),'best_model/model.ckpt')

#ckpt��MNIST
#ckpt1��CIFAR10
#ckpt2��fashionMNIST