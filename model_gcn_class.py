import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd as mmd
import torch.nn.functional as F
import torch

import torch
import os
from torchvision import models
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
# no change
__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class AlexNetFc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """

    def __init__(self, model_path=None, normalize=True):
        super(AlexNetFc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_alexnet = models.alexnet(pretrained=False)
                self.model_alexnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_alexnet = models.alexnet(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_alexnet = self.model_alexnet
        self.features = model_alexnet.features
        self.classifier = nn.Sequential(*list(model_alexnet.classifier.children())[:-1])

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetDSA(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """

    def __init__(self, model_path=None, normalize=True):
        super(AlexNetDSA, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_alexnet = models.alexnet(pretrained=False)
                self.model_alexnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_alexnet = models.alexnet(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.model_alexnet.forward(x)
        return nn.Softmax(dim=-1)(x)

    def output_num(self):
        return self.__in_features


class ResNet50DSA(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """

    def __init__(self, model_path=None, normalize=True):
        super(ResNet50DSA, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.model_resnet.forward(x)
        return nn.Softmax(dim=-1)(x)

    def output_num(self):
        return self.__in_features


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=False):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MDAGM(nn.Module):

    def __init__(self, num_classes=31):
        super(MDAGM, self).__init__()
        self.sharedNet = resnet50(True)

        self.structure_analyzer_1 = model_dict['dsa_alexnet']('D:/cd/CVPGCAN/GCAN/torch_models/alexnet.pth')
        self.structure_analyzer_2 = model_dict['dsa_alexnet']('D:/cd/CVPGCAN/GCAN/torch_models/alexnet.pth')
        self.structure_analyzer_3 = model_dict['dsa_alexnet']('D:/cd/CVPGCAN/GCAN/torch_models/alexnet.pth')

        self.gcns_on1 = GCN(256, 150)
        self.gcns_on2 = GCN(256, 150)
        self.gcns_on3 = GCN(256, 150)

        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.sonnet3 = ADDneck(2048, 256)

        self.cls_fc_son1 = nn.Linear(256 + 150, num_classes)
        self.cls_fc_son2 = nn.Linear(256 + 150, num_classes)
        self.cls_fc_son3 = nn.Linear(256 + 150, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classes = num_classes

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):
        mmd_loss = 0
        intra_loss = 0
        inters_loss = 0
        intert_loss = 0

        if self.training:
            # ---------------------------------------------------------------------------------------
            # 源域数据和目标域数据输入sharedNet 提取common CNN特征
            common_data_src = self.sharedNet(data_src)
            common_data_tgt = self.sharedNet(data_tgt)
            # ---------------------------------------------------------------------------------------
            # 目标域common初级CNN特征 输入第一个源域的Domain-specific Feature Extractor
            # 得到specificCNN特征
            specific_data_tgt_son1 = self.sonnet1(common_data_tgt)
            specific_data_tgt_son1 = self.avgpool(specific_data_tgt_son1)
            feature_target_cnn_on1 = specific_data_tgt_son1.view(specific_data_tgt_son1.size(0), -1)
            # 目标域的数据输入 DSA_on1 提取结构分数特征
            feature_target_structure_on1 = self.structure_analyzer_1.forward(data_tgt)
            adj_target_on1 = torch.mm(feature_target_structure_on1,
                                      torch.transpose(feature_target_structure_on1, 1, 0))
            feature_target_gcn_on1 = self.gcns_on1.forward(feature_target_cnn_on1, adj_target_on1)
            feature_target_on1 = torch.cat([feature_target_cnn_on1, feature_target_gcn_on1], 1)

            pred_tgt_son1 = self.cls_fc_son1(feature_target_on1)
            # ---------------------------------------------------------------------------------------
            specific_data_tgt_son2 = self.sonnet2(common_data_tgt)
            specific_data_tgt_son2 = self.avgpool(specific_data_tgt_son2)
            feature_target_cnn_on2 = specific_data_tgt_son2.view(specific_data_tgt_son2.size(0), -1)

            feature_target_structure_on2 = self.structure_analyzer_2.forward(data_tgt)
            adj_target_on2 = torch.mm(feature_target_structure_on2,
                                      torch.transpose(feature_target_structure_on2, 1, 0))
            feature_target_gcn_on2 = self.gcns_on2.forward(feature_target_cnn_on2, adj_target_on2)
            feature_target_on2 = torch.cat([feature_target_cnn_on2, feature_target_gcn_on2], 1)

            pred_tgt_son2 = self.cls_fc_son2(feature_target_on2)
            # ---------------------------------------------------------------------------------------
            specific_data_tgt_son3 = self.sonnet3(common_data_tgt)
            specific_data_tgt_son3 = self.avgpool(specific_data_tgt_son3)
            feature_target_cnn_on3 = specific_data_tgt_son3.view(specific_data_tgt_son3.size(0), -1)

            feature_target_structure_on3 = self.structure_analyzer_3.forward(data_tgt)
            adj_target_on3 = torch.mm(feature_target_structure_on3,
                                      torch.transpose(feature_target_structure_on3, 1, 0))
            feature_target_gcn_on3 = self.gcns_on3.forward(feature_target_cnn_on3, adj_target_on3)
            feature_target_on3 = torch.cat([feature_target_cnn_on3, feature_target_gcn_on3], 1)

            pred_tgt_son3 = self.cls_fc_son3(feature_target_on3)

            if mark == 1:
                # 第一个源域common初级CNN特征 输入自己的Domain-specific Feature Extractor
                # 得到specificCNN特征
                specific_data_src = self.sonnet1(common_data_src)
                specific_data_src = self.avgpool(specific_data_src)
                feature_source_cnn = specific_data_src.view(specific_data_src.size(0), -1)

                # 第一个源域的数据输入 DSA_on1 提取结构分数特征
                feature_source_structure = self.structure_analyzer_1.forward(data_src)
                adj_source = torch.mm(feature_source_structure,
                                      torch.transpose(feature_source_structure, 1, 0))

                feature_source_gcn = self.gcns_on1.forward(feature_source_cnn, adj_source)
                feature_source = torch.cat([feature_source_cnn, feature_source_gcn], 1)
                # ---------------------------------------------------------------------------------------
                mmd_loss += mmd.mmd(feature_source, feature_target_on1)

                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son1(feature_source)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                s_label = self.softmax(pred_src)

                t_label = self.softmax(pred_tgt_son1)

                sums_label = s_label.data.sum(0)
                sumt_label = t_label.data.sum(0)
                smax = sums_label.data.max(0)[1]
                tmax = sumt_label.data.max(0)[1]
                sums_label[smax] = 0
                sumt_label[tmax] = 0

                smax2 = sums_label.data.max(0)[1]
                tmax2 = sumt_label.data.max(0)[1]

                for c in range(self.classes):
                    ps = s_label[:, c].reshape(feature_source.shape[0], 1)
                    pt = t_label[:, c].reshape(feature_source.shape[0], 1)
                    intra_loss += mmd.mmd(ps * feature_source, pt * feature_target_on1)

                ps1 = s_label[:, smax].reshape(feature_source.shape[0], 1)
                ps2 = s_label[:, smax2].reshape(feature_source.shape[0], 1)
                inters_loss += mmd.mmd(ps1 * feature_source, ps2 * feature_source)

                pt1 = t_label[:, tmax].reshape(feature_source.shape[0], 1)
                pt2 = t_label[:, tmax2].reshape(feature_source.shape[0], 1)
                intert_loss += mmd.mmd(pt1 * feature_target_on1, pt2 * feature_target_on1)

                class_loss = intra_loss / self.classes - 0.01 * (inters_loss + intert_loss) / 2

                return cls_loss, mmd_loss, l1_loss / 2, class_loss

            if mark == 2:
                specific_data_src = self.sonnet2(common_data_src)
                specific_data_src = self.avgpool(specific_data_src)
                feature_source_cnn = specific_data_src.view(specific_data_src.size(0), -1)

                # 第一个源域的数据输入 DSA_on1 提取结构分数特征
                feature_source_structure = self.structure_analyzer_2.forward(data_src)
                adj_source = torch.mm(feature_source_structure,
                                      torch.transpose(feature_source_structure, 1, 0))

                feature_source_gcn = self.gcns_on2.forward(feature_source_cnn, adj_source)
                feature_source = torch.cat([feature_source_cnn, feature_source_gcn], 1)
                # ---------------------------------------------------------------------------------------
                mmd_loss += mmd.mmd(feature_source, feature_target_on2)

                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son2(feature_source)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                s_label = self.softmax(pred_src)
                t_label = self.softmax(pred_tgt_son2)

                sums_label = s_label.data.sum(0)
                sumt_label = t_label.data.sum(0)
                smax = sums_label.data.max(0)[1]
                tmax = sumt_label.data.max(0)[1]
                sums_label[smax] = 0
                sumt_label[tmax] = 0

                smax2 = sums_label.data.max(0)[1]
                tmax2 = sumt_label.data.max(0)[1]

                for c in range(self.classes):
                    ps = s_label[:, c].reshape(feature_source.shape[0], 1)
                    pt = t_label[:, c].reshape(feature_source.shape[0], 1)
                    intra_loss += mmd.mmd(ps * feature_source, pt * feature_target_on2)

                ps1 = s_label[:, smax].reshape(feature_source.shape[0], 1)
                ps2 = s_label[:, smax2].reshape(feature_source.shape[0], 1)
                inters_loss += mmd.mmd(ps1 * feature_source, ps2 * feature_source)

                pt1 = t_label[:, tmax].reshape(feature_source.shape[0], 1)
                pt2 = t_label[:, tmax2].reshape(feature_source.shape[0], 1)
                intert_loss += mmd.mmd(pt1 * feature_target_on2, pt2 * feature_target_on2)

                class_loss = intra_loss / self.classes - 0.01 * (inters_loss + intert_loss) / 2

                return cls_loss, mmd_loss, l1_loss / 2, class_loss

            if mark == 3:
                specific_data_src = self.sonnet3(common_data_src)
                specific_data_src = self.avgpool(specific_data_src)
                feature_source_cnn = specific_data_src.view(specific_data_src.size(0), -1)

                # 第一个源域的数据输入 DSA_on1 提取结构分数特征
                feature_source_structure = self.structure_analyzer_3.forward(data_src)
                adj_source = torch.mm(feature_source_structure,
                                      torch.transpose(feature_source_structure, 1, 0))

                feature_source_gcn = self.gcns_on3.forward(feature_source_cnn, adj_source)
                feature_source = torch.cat([feature_source_cnn, feature_source_gcn], 1)
                # ---------------------------------------------------------------------------------------
                mmd_loss += mmd.mmd(feature_source, feature_target_on3)

                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                pred_src = self.cls_fc_son3(feature_source)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                s_label = self.softmax(pred_src)
                t_label = self.softmax(pred_tgt_son3)

                sums_label = s_label.data.sum(0)
                sumt_label = t_label.data.sum(0)
                smax = sums_label.data.max(0)[1]
                tmax = sumt_label.data.max(0)[1]
                sums_label[smax] = 0
                sumt_label[tmax] = 0

                smax2 = sums_label.data.max(0)[1]
                tmax2 = sumt_label.data.max(0)[1]

                for c in range(self.classes):
                    ps = s_label[:, c].reshape(feature_source.shape[0], 1)
                    pt = t_label[:, c].reshape(feature_source.shape[0], 1)
                    intra_loss += mmd.mmd(ps * feature_source, pt * feature_target_on3)

                ps1 = s_label[:, smax].reshape(feature_source.shape[0], 1)
                ps2 = s_label[:, smax2].reshape(feature_source.shape[0], 1)
                inters_loss += mmd.mmd(ps1 * feature_source, ps2 * feature_source)

                pt1 = t_label[:, tmax].reshape(feature_source.shape[0], 1)
                pt2 = t_label[:, tmax2].reshape(feature_source.shape[0], 1)
                intert_loss += mmd.mmd(pt1 * feature_target_on3, pt2 * feature_target_on3)

                class_loss = intra_loss / self.classes - 0.01 * (inters_loss + intert_loss) / 2

                return cls_loss, mmd_loss, l1_loss / 2, class_loss

        else:
            data = self.sharedNet(data_src)

            structure_score_on1 = self.structure_analyzer_1(data_src)
            structure_score_on2 = self.structure_analyzer_2(data_src)
            structure_score_on3 = self.structure_analyzer_3(data_src)
            adj_on1 = torch.mm(structure_score_on1, torch.transpose(structure_score_on1, 1, 0))
            adj_on2 = torch.mm(structure_score_on2, torch.transpose(structure_score_on2, 1, 0))
            adj_on3 = torch.mm(structure_score_on3, torch.transpose(structure_score_on3, 1, 0))

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            gcn_on1 = self.gcns_on1(fea_son1, adj_on1)
            feature_1 = torch.cat([fea_son1, gcn_on1], 1)

            pred1 = self.cls_fc_son1(feature_1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            gcn_on2 = self.gcns_on2(fea_son2, adj_on2)
            feature_2 = torch.cat([fea_son2, gcn_on2], 1)

            pred2 = self.cls_fc_son2(feature_2)

            fea_son3 = self.sonnet3(data)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            gcn_on3 = self.gcns_on3(fea_son3, adj_on3)
            feature_3 = torch.cat([fea_son3, gcn_on3], 1)

            pred3 = self.cls_fc_son3(feature_3)

            return pred1, pred2, pred3


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


model_dict = {

    'dsa_resnet50': ResNet50DSA,
    'dsa_alexnet': AlexNetDSA
}
