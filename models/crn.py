
'''
Author: Fakai Wang
Affiliation: Ruijn Hospital, Shanghai Jiao Tong University School of Medicine

'''
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
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


CRN_cfg = {
    'CRNfull0.1': {'channels': [4, 4, 6, 6, 8, 8], 'num_blocks': [2, 2, 2, 2], 'block': BasicBlock},
    'CRNfull0.2': {'channels': [4, 8, 8, 8, 8, 8], 'num_blocks': [2, 2, 2, 2], 'block': BasicBlock},
    'CRNfull0.3': {'channels': [4, 8, 8, 8, 8, 16], 'num_blocks': [2, 2, 2, 2], 'block': BasicBlock},

    'CRNfull1.1': {'channels': [4, 8, 8, 8, 16, 32], 'num_blocks': [2, 2, 2, 2], 'block': BasicBlock},
    'CRNfull1.2': {'channels': [4, 8, 8, 8, 16, 32], 'num_blocks': [3, 4, 6, 3], 'block': BasicBlock},
    'CRNfull1.3': {'channels': [4, 16, 16, 32, 32, 64], 'num_blocks':[2, 2, 2, 2], 'block':BasicBlock},
    'CRNfull1.4': {'channels': [4, 16, 16, 32, 32, 64], 'num_blocks':[3, 4, 6, 3], 'block':BasicBlock},
    'CRNfull1.5': {'channels': [4, 8, 8, 16, 128, 128], 'num_blocks': [3, 4, 6, 3], 'block': Bottleneck},
}

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CRN(nn.Module):
    def __init__(self, resnet_name, in_modality=10, in_channels=4, target_classes=None, net_configs=None, 
                     return_feats=True):
        super(CRN, self).__init__()
        self.in_modality = in_modality
        self.in_channels = in_channels
        self.signal_dependencies = net_configs['signal_dependencies']
        self.target_classes = target_classes
        self.return_feats = return_feats
        self.block = CRN_cfg[resnet_name].get('block', BasicBlock)
        self.channels = CRN_cfg[resnet_name]['channels']
        self.channels[0] = in_channels
        self.num_blocks = CRN_cfg[resnet_name]['num_blocks']
        self.sep_layers = nn.ModuleList([self._sep_layers() for i in range(self.in_modality)])
        self.joint_layers = nn.ModuleList(self._joint_layers(self.signal_dependencies))
        self.N_classifier = len(target_classes)
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.channels[-1] * self.block.expansion, target_classes[i]) for i in range(self.N_classifier)])
        total_signals = sum([target_classes[i] for i in range(self.N_classifier)])
        self.classifier_disease = nn.Linear(total_signals, target_classes[-1])

        dim = self.channels[-1] * self.block.expansion
        drop_path = 0.2
        layer_scale = False
        init_value = 1e-6
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=nn.GELU, drop=0.)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

    def _sep_layers(self):
        block = self.block
        num_blocks = self.num_blocks
        channels = self.channels
        input_layer = InputBlock(channels[0], channels[1])
        self.in_planes = channels[1]
        layer1 = self._make_layer(block, channels[2], num_blocks[0], stride=1)
        layer2 = self._make_layer(block, channels[3], num_blocks[1], stride=2)
        return nn.Sequential(input_layer, layer1, layer2)

    def _joint_layers(self, signal_dependencies):
        block = self.block
        num_blocks = self.num_blocks
        channels = self.channels
        joint_layers = []
        self.in_planes_middle = self.in_planes # save the self.in_planes right after sep layers (1 & 2)
        for mod_ids in signal_dependencies:
            self.in_planes = self.in_planes_middle * len(mod_ids)
            layer3 = self._make_layer(block, channels[4], num_blocks[2], stride=2)
            layer4 = self._make_layer(block, channels[5], num_blocks[3], stride=2)
            joint_layers.append(nn.Sequential(layer3, layer4))
        return joint_layers

    def _com_layers(self):
        block = self.block
        num_blocks = self.num_blocks
        channels = self.channels
        self.in_planes = self.in_planes * self.in_modality # common layers: concatenate all the features
        layer3 = self._make_layer(block, channels[4], num_blocks[2], stride=2)
        layer4 = self._make_layer(block, channels[5], num_blocks[3], stride=2)
        return nn.Sequential(layer3, layer4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], self.in_modality, self.in_channels, shape[2], shape[3])
        outs = [self.sep_layers[i](x[:, i, :, :, :]) for i in range(self.in_modality)]
        features_all = []
        outputs = []
        for i in range(self.N_classifier):
            feat_comb = [outs[j] for j in self.signal_dependencies[i]]
            feat_A = torch.cat(feat_comb, 1)  #
            output = self.joint_layers[i](feat_A)
            output = F.avg_pool2d(output, 4)
            output = F.adaptive_avg_pool2d(output, (1, 1))
            features = output.view(output.size(0), 1, -1)
            features_all.append(features) # TODO: add attention mechanism here, before classification
        x = torch.cat(features_all, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        for i in range(self.N_classifier):
            output = self.classifiers[i](x[:, i, :])
            outputs.append(output)
        feat_comb = [F.sigmoid(outputs[i]) for i in range(self.N_classifier - 1)]
        feat_comb.extend([F.relu(outputs[self.N_classifier - 1])])
        features = torch.cat(feat_comb, 1)  #
        output = self.classifier_disease(features)
        outputs[-1] = output

        if self.return_feats:
            return x, outputs
        else:
            return outputs
