'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG1.1': [8, 'M', 16, 'M', 16, 8, 'M', 8, 'M'],
    'VGG1.2': [16, 'M', 8, 'M', 16, 8, 'M', 4, 'M'],
    'VGG1.3': [16, 'M', 32, 'M', 16, 16, 'M', 8, 'M'],
    'VGG1.4': [16, 'M', 16, 'M', 16, 16, 'M', 8, 'M'],
    'VGG1.5': [32, 'M', 16, 'M', 32, 'M', 16, 'M'],
    'VGG1.6': [32, 'M', 16, 'M', 32, 'M', 32, 16, 'M'],
    'VGG1.7': [32, 'M', 16, 'M', 32, 32, 'M', 32, 16, 'M'],
    'VGG1.8': [32, 'M', 64, 'M', 64, 64, 'M', 64, 32, 'M'],
    'VGG1.9': [32, 'M', 32, 'M', 32, 32, 'M', 32, 32, 'M'],
}

cfg.update({
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
})


class VGG_aux(nn.Module): # EF: Early Fusion, default
    def __init__(self, net_name, in_modality=10, in_channels=4, target_classes=None, return_feats=True):
        super(VGG_aux, self).__init__()
        self.return_feats = return_feats
        self.in_channels = in_modality * in_channels
        print(f'VGG:self.in_channels ={self.in_channels }')
        self.features = self._make_layers(cfg[net_name])
        self.N_classifier = len(target_classes)
        self.classifiers = nn.ModuleList(
            [nn.Linear(cfg[net_name][-2], target_classes[i]) for i in range(self.N_classifier)])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        features_all = out
        outputs = [self.classifiers[i](out) for i in range(len(self.classifiers))]
        if self.return_feats:
            return features_all, outputs
        else:
            return outputs

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)



class VGG_relay(nn.Module):
    def __init__(self, vgg_name, in_modality = 10, in_channels = 4, signal_modalities=None, target_classes=None, return_feats = True):
        print('VGG_aux:', vgg_name, cfg[vgg_name])
        super(VGG_relay, self).__init__()
        self.in_modality = in_modality
        self.in_channels = in_channels
        self.signal_modalities = signal_modalities
        self.target_classes = target_classes
        self.return_feats = return_feats
        self.features = nn.ModuleList([self._make_layers(cfg[vgg_name], in_channels) for i in range(in_modality)])
        # self.drop = nn.Dropout(0.8)
        self.N_classifier = len(target_classes)
        N_Feat = cfg[vgg_name][-2]
        self.classifiers = nn.ModuleList([nn.Linear(N_Feat * len(signal_modalities[i]), target_classes[i]) for i in range(self.N_classifier)])
        total_signals = sum([target_classes[i] for i in range(self.N_classifier)])
        self.classifier_disease = nn.Linear(total_signals, target_classes[-1])

    def forward_simple(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(shape[0], self.in_modality, self.in_channels, shape[2], shape[3])
        outs = [self.features[i](x[:, i, :, :, :]) for i in range(self.in_modality)]
        outs = [out.view(out.size(0), -1) for out in outs]
        features_all = torch.cat(outs, 1)  #
        outputs = []
        for i in range(self.N_classifier):
            feat_comb = [outs[j] for j in self.signal_modalities[i]]
            features = torch.cat(feat_comb, 1)  #
            output = self.classifiers[i](features)
            outputs.append(output)

        if self.return_feats:
            return features_all, outputs
        else:
            return outputs

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(shape[0], self.in_modality, self.in_channels, shape[2], shape[3])
        outs = [self.features[i](x[:, i, :, :, :]) for i in range(self.in_modality)]
        outs = [out.view(out.size(0), -1) for out in outs]
        features_all = torch.cat(outs, 1)  #
        outputs = []
        for i in range(self.N_classifier):
            feat_comb = [outs[j] for j in self.signal_modalities[i]]
            features = torch.cat(feat_comb, 1)  #
            output = self.classifiers[i](features)
            outputs.append(output)

        feat_comb = [F.sigmoid(outputs[i]) for i in range(self.N_classifier - 1)]
        feat_comb.extend([F.relu(outputs[self.N_classifier - 1])])
        features = torch.cat(feat_comb, 1)  #
        output = self.classifier_disease(features)
        outputs[-1] = output


        if self.return_feats:
            return features_all, outputs
        else:
            return outputs

    def _make_layers(self, cfg, in_channels = 2):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)
