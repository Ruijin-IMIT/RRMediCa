'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


class InputBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(InputBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print('InputBlock out.shape:', out.shape)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print('BasicBlock x.shape:', x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print('BasicBlock out.shape:', out.shape)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

resnet_cfg = {
    'resnet18': {'channels':[4, 64, 64, 128, 256, 512], 'num_blocks':[2, 2, 2, 2], 'block':BasicBlock},
    'resnet34': {'channels':[4, 64, 64, 128, 256, 512], 'num_blocks':[3, 4, 6, 3], 'block':BasicBlock},
    'resnet50': {'channels':[4, 64, 64, 128, 256, 512], 'num_blocks':[3, 4, 6, 3], 'block':Bottleneck},

    'resnet1.1': {'channels': [4, 8, 8, 8, 16, 32], 'num_blocks': [2, 2, 2, 2], 'block': BasicBlock},
    'resnet1.2': {'channels': [4, 8, 8, 8, 16, 32], 'num_blocks': [3, 4, 6, 3], 'block': BasicBlock},
    'resnet1.3': {'channels': [4, 16, 16, 32, 32, 64], 'num_blocks':[2, 2, 2, 2], 'block':BasicBlock},
    'resnet1.4': {'channels': [4, 16, 16, 32, 32, 64], 'num_blocks':[3, 4, 6, 3], 'block':BasicBlock},
    'resnet1.5': {'channels': [4, 8, 8, 16, 128, 128], 'num_blocks': [3, 4, 6, 3], 'block': Bottleneck},
}


class ResNet_aux(nn.Module):
    def __init__(self, net_name, in_modality=10, in_channels=4, target_classes=None, return_feats=True):
        super(ResNet_aux, self).__init__()
        self.return_feats = return_feats
        self.block = resnet_cfg[net_name].get('block', BasicBlock)
        # self.channels = [4, 32, 32, 64, 64, 64]
        # self.num_blocks = [2, 2, 2, 2]
        self.channels = resnet_cfg[net_name]['channels']
        self.channels[0] = in_modality * in_channels
        self.num_blocks = resnet_cfg[net_name]['num_blocks']

        self.in_planes = self.channels[1]
        self.conv1 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[1])
        self.layer1 = self._make_layer(self.block, self.channels[2], self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, self.channels[3], self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, self.channels[4], self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, self.channels[5], self.num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*self.block.expansion, num_classes)
        self.N_classifier = len(target_classes)
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.channels[5]*self.block.expansion, target_classes[i]) for i in range(self.N_classifier)])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        features_all = out
        outputs = [self.classifiers[i](out) for i in range(len(self.classifiers))]
        if self.return_feats:
            return features_all, outputs
        else:
            return outputs

