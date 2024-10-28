import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                     'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # original padding is 1; original dilation is 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                     padding=dilation, bias=False, dilation=dilation)


class Sequential(nn.Sequential):
        # def __init__(self, *args):
        #   super(Sequential, self).__init__()
    # 重写sequential类，每一层都可以根据modal参数进行不同的计算
    def forward(self, input, modal=0):
        # 循环遍历Sequential中包含的每一个层（每一个module是一个子模块）
        for module in self:
            res = module(input, modal)
            input = res
        return res

# ResNet-18 和 ResNet-34 使用的 BasicBlock 结构较为简单，主要用于提取低层特征，这些低层特征在不同模态间共享性较强，因此不需要为每个模态设定不同的 BatchNorm 层。
# 模态的差异 更多体现在高层特征上，而 BasicBlock 中提取的低级特征通常是模态无关的。
# 对于较深的模型如 ResNet-50 及以上，使用 Bottleneck 块处理更高级别的特征，因此在这些更复杂的结构中为不同模态引入独立的 BatchNorm 层是必要的。
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        # 指定了使用 2D 批归一化
        # BatchNorm2d对每一批数据中的每个通道，计算其均值和标准差。
        # 使用计算出的均值和标准差将每个通道的数据归一化为零均值和单位方差。
        # 引入可学习的尺度参数（gamma）和偏移参数（beta），对归一化后的数据进行线性变换
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
        # 如果卷积操作改变了空间尺寸或者通道数发生变化，那么downsample就会被用来调整输入特征图的尺寸和通道数。
        # downsample通常是通过一个卷积操作或者平均池化操作来实现。
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, isshape=False, modalbn=1):
        super(Bottleneck, self).__init__()

        self.isshape = isshape
        self.modalbn = modalbn
        assert modalbn == 1 or modalbn == 2 or modalbn == 3

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # improved for diverse modality 对不同图像进行不同处理
        # 依据模态的不同设置不同的bn1
        if isshape:
            self.bn1_shape = nn.BatchNorm2d(planes)
        if modalbn == 2:
            self.bn1_ir = nn.BatchNorm2d(planes)
        if modalbn == 3:
            self.bn1_ir = nn.BatchNorm2d(planes)
            self.bn1_modalx = nn.BatchNorm2d(planes)

        # original padding is 1; original dilation is 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        # 依据模态的不同设置不同的bn2
        if isshape:
            self.bn2_shape = nn.BatchNorm2d(planes)
        if modalbn == 2:
            self.bn2_ir = nn.BatchNorm2d(planes)
        if modalbn == 3:
            self.bn2_ir = nn.BatchNorm2d(planes)
            self.bn2_modalx = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # 依据模态的不同设置不同的bn3
        if isshape:
            self.bn3_shape = nn.BatchNorm2d(planes * 4)
        if modalbn == 2:
            self.bn3_ir = nn.BatchNorm2d(planes * 4)
        if modalbn == 3:
            self.bn3_ir = nn.BatchNorm2d(planes * 4)
            self.bn3_modalx = nn.BatchNorm2d(planes * 4)

        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # 依据模态的不同设置不同的downsample的BN
        if downsample is not None:
            if isshape:
                self.dsbn_shape = nn.BatchNorm2d(downsample[1].weight.shape[0])
            if modalbn == 2:
                self.dsbn_ir = nn.BatchNorm2d(downsample[1].weight.shape[0])
            if modalbn == 3:
                self.dsbn_ir = nn.BatchNorm2d(downsample[1].weight.shape[0])
                self.dsbn_modalx = nn.BatchNorm2d(downsample[1].weight.shape[0])
        self.stride = stride

    def forward(self, x, modal=0):
        if modal == 0: # RGB
            bbn1 = self.bn1
            bbn2 = self.bn2
            bbn3 = self.bn3
            if self.downsample is not None:
                dsbn = self.downsample[1]
        elif modal == 1: # IR
            bbn1 = self.bn1_ir
            bbn2 = self.bn2_ir
            bbn3 = self.bn3_ir
            if self.downsample is not None:
                dsbn = self.dsbn_ir
        elif modal == 2: # modalx
            bbn1 = self.bn1_modalx
            bbn2 = self.bn2_modalx
            bbn3 = self.bn3_modalx
            if self.downsample is not None:    
                dsbn = self.dsbn_modalx
        elif modal == 3: # shape
            assert self.isshape == True
            bbn1 = self.bn1_shape
            bbn2 = self.bn2_shape
            bbn3 = self.bn3_shape
            if self.downsample is not None:
                dsbn = self.dsbn_shape

        residual = x
        # 根据模态的不同设定不同的bbn（BatchNormalization)
        out = self.conv1(x)
        out = bbn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = bbn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = bbn3(out)
        # downsample[0]：这是一个1x1的卷积层(nn.Conv2d)。
        # downsample[1]：这是对应的批归一化层(nn.BatchNorm2d)。
        if self.downsample is not None:
                # 先进行卷积调整通道，再进行归一化
                residual = dsbn(self.downsample[0](x))

        out += residual
        outt = F.relu(out)

        return outt


class ResNet(nn.Module):

    def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1, isshape=False, modalbn=1, onlyshallow=False):
        self.isshape = isshape
        self.modalbn = modalbn
        self.inplanes = 64
        super(ResNet, self).__init__()
        # if onlyshallow:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                                    bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if self.isshape:
            self.bn1_shape = nn.BatchNorm2d(64)
        if self.modalbn == 2:
            self.bn1_ir = nn.BatchNorm2d(64)
        elif self.modalbn == 3:
            self.bn1_ir = nn.BatchNorm2d(64)
            self.bn1_modalx = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # else:
        # 通过_make_layer带到层次化设计的效果
        self.layer1 = self._make_layer(block, 64, layers[0]) # 对应着conv2_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 对应着conv3_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 对应着conv4_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation) # 对应着conv5_x

        # 对模型的每一个模块的参数，根据模块的类型设定参数的初始化
        # 卷积层的初始化：使用Kaiming初始化，根据卷积核的大小和输出通道数动态调整权重的标准差，保证深层网络的梯度稳定传播。
        # 批归一化层的初始化：权重初始化为1，偏置初始化为0，这样保证了在训练开始时，特征不会被过多修改，有助于网络的初始收敛
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 层次化设计，每一个_make_layer构造一层残差块，blocks就是每一块调用的block(basicblock或者bottenneck)次数，对于每一次调用的block，首先设定downsample，然后用layers将blocks次组装起来成一个残差块
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        # 设置downsample，若expansion不等于1或者是stride不等于1，则需要对shortcut进行downsample处理，使得downsaple后的x能与经过残差块后的结果相加
        # 使用conv1*1，减少通道数，并且设定stride，若stride不为1，则同样设定conv1*1的stride，使宽度高度相同
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # 对于第一次第一次调用的block,inplane,plane分为输入输出的通道数，然后根据expansion可以得出输出的通道数，即后面每一次输入的inplane，就是planes * block.expansion
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, isshape=self.isshape, modalbn=self.modalbn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, isshape=self.isshape, modalbn=self.modalbn))
        #使用Sequential构建网络，将各层拼起来形成ResNet
        return Sequential(*layers)

    def forward(self, x, modal=0):
        x = self.conv1(x)
        if modal == 0: # RGB
            bbn1 = self.bn1
        elif modal == 1: # IR
            bbn1 = self.bn1_ir
        elif modal == 2: # modalx
            bbn1 = self.bn1_modalx
        elif modal == 3: # shape
            assert self.isshape == True
            bbn1 = self.bn1_shape

        x = bbn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, modal)
        x = self.layer2(x, modal)
        x = self.layer3(x, modal)
        x = self.layer4(x, modal)

        return x

    def init_bn_layer(self, layer):
        for i in range(len(layer)):
            if self.isshape:
                layer[i].bn1_shape.weight.data = layer[i].bn1.weight.data.clone()
                layer[i].bn1_shape.bias.data = layer[i].bn1.bias.data.clone()
                layer[i].bn2_shape.weight.data = layer[i].bn2.weight.data.clone()
                layer[i].bn2_shape.bias.data = layer[i].bn2.bias.data.clone()
                layer[i].bn3_shape.weight.data = layer[i].bn3.weight.data.clone()
                layer[i].bn3_shape.bias.data = layer[i].bn3.bias.data.clone()
                if layer[i].downsample is not None:
                    layer[i].dsbn_shape.weight.data = layer[i].downsample[1].weight.data.clone()
                    layer[i].dsbn_shape.bias.data = layer[i].downsample[1].bias.data.clone()
            if self.modalbn >= 2:
                layer[i].bn1_ir.weight.data = layer[i].bn1.weight.data.clone()
                layer[i].bn1_ir.bias.data = layer[i].bn1.bias.data.clone()
                layer[i].bn2_ir.weight.data = layer[i].bn2.weight.data.clone()
                layer[i].bn2_ir.bias.data = layer[i].bn2.bias.data.clone()
                layer[i].bn3_ir.weight.data = layer[i].bn3.weight.data.clone()
                layer[i].bn3_ir.bias.data = layer[i].bn3.bias.data.clone()
                if layer[i].downsample is not None:
                    layer[i].dsbn_ir.weight.data = layer[i].downsample[1].weight.data.clone()
                    layer[i].dsbn_ir.bias.data = layer[i].downsample[1].bias.data.clone()
            if self.modalbn == 3:
                layer[i].bn1_modalx.weight.data = layer[i].bn1.weight.data.clone()
                layer[i].bn1_modalx.bias.data = layer[i].bn1.bias.data.clone()
                layer[i].bn2_modalx.weight.data = layer[i].bn2.weight.data.clone()
                layer[i].bn2_modalx.bias.data = layer[i].bn2.bias.data.clone()
                layer[i].bn3_modalx.weight.data = layer[i].bn3.weight.data.clone()
                layer[i].bn3_modalx.bias.data = layer[i].bn3.bias.data.clone()
                if layer[i].downsample is not None:
                    layer[i].dsbn_modalx.weight.data = layer[i].downsample[1].weight.data.clone()
                    layer[i].dsbn_modalx.bias.data = layer[i].downsample[1].bias.data.clone()
    # 对BN层进行参数的初始化
    def init_bn(self, onlyshallow=False):
        # if onlyshallow:
        if self.isshape:
            self.bn1_shape.weight.data = self.bn1.weight.data.clone()
            self.bn1_shape.bias.data = self.bn1.bias.data.clone()
        if self.modalbn >= 2:
            self.bn1_ir.weight.data = self.bn1.weight.data.clone()
            self.bn1_ir.bias.data = self.bn1.bias.data.clone()
        if self.modalbn == 3:
            self.bn1_modalx.weight.data = self.bn1.weight.data.clone()
            self.bn1_modalx.bias.data = self.bn1.bias.data.clone()
        # else:
        self.init_bn_layer(self.layer1)
        self.init_bn_layer(self.layer2)
        self.init_bn_layer(self.layer3)
        self.init_bn_layer(self.layer4)

    # layer[i].bnx(x=1,2,3)共享多个模态的归一化参数weight和bias
    def average_bn_layer(self, layer):
        for i in range(len(layer)):
            bn1w = (layer[i].bn1.weight.data.clone()+layer[i].bn1_ir.weight.data.clone()+layer[i].bn1_shape.weight.data.clone())/3
            bn1b = (layer[i].bn1.bias.data.clone()+layer[i].bn1_ir.bias.data.clone()+layer[i].bn1_shape.bias.data.clone())/3
            bn2w = (layer[i].bn2.weight.data.clone()+layer[i].bn2_ir.weight.data.clone()+layer[i].bn2_shape.weight.data.clone())/3
            bn2b = (layer[i].bn2.bias.data.clone()+layer[i].bn2_ir.bias.data.clone()+layer[i].bn2_shape.bias.data.clone())/3
            bn3w = (layer[i].bn3.weight.data.clone()+layer[i].bn3_ir.weight.data.clone()+layer[i].bn3_shape.weight.data.clone())/3
            bn3b = (layer[i].bn3.bias.data.clone()+layer[i].bn3_ir.bias.data.clone()+layer[i].bn3_shape.bias.data.clone())/3
            if layer[i].downsample is not None:
                dbbnw = (layer[i].downsample[1].weight.data.clone()+layer[i].dsbn_shape.weight.data.clone()+layer[i].dsbn_ir.weight.data.clone())/3
                dbbnb = (layer[i].downsample[1].bias.data.clone()+layer[i].dsbn_shape.bias.data.clone()+layer[i].dsbn_ir.bias.data.clone())/3
            layer[i].bn1.weight.data = bn1w.clone()
            layer[i].bn1.bias.data = bn1b.clone()
            layer[i].bn2.weight.data = bn2w.clone()
            layer[i].bn2.bias.data = bn2b.clone()
            layer[i].bn3.weight.data = bn3w.clone()
            layer[i].bn3.bias.data = bn3b.clone()
            if layer[i].downsample is not None:
                layer[i].downsample[1].weight.data = dbbnw.clone()
                layer[i].downsample[1].bias.data = dbbnb.clone()
            if self.isshape:
                layer[i].bn1_shape.weight.data = bn1w.clone()
                layer[i].bn1_shape.bias.data = bn1b.clone()
                layer[i].bn2_shape.weight.data = bn2w.clone()
                layer[i].bn2_shape.bias.data = bn2b.clone()
                layer[i].bn3_shape.weight.data = bn3w.clone()
                layer[i].bn3_shape.bias.data = bn3b.clone()
                if layer[i].downsample is not None:
                    layer[i].dsbn_shape.weight.data = dbbnw.clone()
                    layer[i].dsbn_shape.bias.data = dbbnb.clone()
            if self.modalbn >= 2:
                layer[i].bn1_ir.weight.data = bn1w.clone()
                layer[i].bn1_ir.bias.data = bn1b.clone()
                layer[i].bn2_ir.weight.data = bn2w.clone()
                layer[i].bn2_ir.bias.data = bn2b.clone()
                layer[i].bn3_ir.weight.data = bn3w.clone()
                layer[i].bn3_ir.bias.data = bn3b.clone()
                if layer[i].downsample is not None:
                    layer[i].dsbn_ir.weight.data = dbbnw.clone()
                    layer[i].dsbn_ir.bias.data = dbbnb.clone()
            if self.modalbn == 3:
                layer[i].bn1_modalx.weight.data = bn1w.clone()
                layer[i].bn1_modalx.bias.data = bn1b.clone()
                layer[i].bn2_modalx.weight.data = bn2w.clone()
                layer[i].bn2_modalx.bias.data = bn2b.clone()
                layer[i].bn3_modalx.weight.data = bn3w.clone()
                layer[i].bn3_modalx.bias.data = bn3b.clone()
                if layer[i].downsample is not None:
                    layer[i].dsbn_modalx.weight.data = dbbnw.clone()
                    layer[i].dsbn_modalx.bias.data = dbbnb.clone()
    # 对每一个残差块进行多模态的参数共享，根据onlyshallow及逆行不同的参数设置，若为onlyshallow则只对bn1进行共享，否则对layer1,2,3,4都进行共享
    def average_bn(self, onlyshallow=False):
        if onlyshallow:
            tmpw = self.bn1.weight.data.clone() + self.bn1_shape.weight.data.clone()
            tmpw /= 2
            tmpb = self.bn1.bias.data.clone() + self.bn1_shape.bias.data.clone()
            tmpb /= 2
            self.bn1.weight.data = tmpw.clone()
            self.bn1.bias.data = tmpb.clone()

            if self.isshape:
                self.bn1_shape.weight.data = tmpw.clone()
                self.bn1_shape.bias.data = tmpb.clone()
            if self.modalbn >= 2:
                self.bn1_ir.weight.data = tmpw.clone()
                self.bn1_ir.bias.data = tmpb.clone()
            if self.modalbn == 3:
                self.bn1_modalx.weight.data = tmpw.clone()
                self.bn1_modalx.bias.data = tmpb.clone()
        else:
            self.average_bn_layer(self.layer1)
            self.average_bn_layer(self.layer2)
            self.average_bn_layer(self.layer3)
            self.average_bn_layer(self.layer4)
    # def last_layer_shared(self, ):
    #     res = []
    #     for k, v in self.layer4.named_parameters():
    #         if 'conv' in k:
    #             res.append(v)
    #     return res
# 从模型的 state_dict 中移除全连接层（fc层）的参数。因为在某些情况下（如迁移学习或多模态任务），我们可能不需要预训练模型的全连接层，而是只需要卷积层的权重
# state_dict：这是模型的所有参数和缓冲区的字典。每个参数和缓冲区以键值对的形式存储。
# 删除全连接层的参数后，可以在不依赖原来的分类头（全连接层）的情况下，使用预训练模型的特征提取部分（卷积层等）
def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    # for key, value in state_dict.items():
    for key, value in list(state_dict.items()):
        if key.startswith('fc.'):
            del state_dict[key]
    return state_dict

# 各个 ResNet 构造函数
# 如果 pretrained=True，表示使用在 ImageNet 数据集上训练好的模型权重。
# 通过 model_zoo.load_url(model_urls['resnetx']) 从 URL 下载预训练权重，并通过 remove_fc 删除全连接层的参数，然后加载到模型中。
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
    return model

# strict=False，允许模型加载部分权重，并忽略那些不匹配的部分。
# 这非常适用于重写后的 ResNet，因为它包含了额外的模态特定的 BatchNorm 层，而这些层在预训练模型中是不存在的。通过 strict=False，模型会跳过这些层的权重加载，而不会导致错误
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])),strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            remove_fc(model_zoo.load_url(model_urls['resnet101'])),strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            remove_fc(model_zoo.load_url(model_urls['resnet152'])), strict=False)
    return model