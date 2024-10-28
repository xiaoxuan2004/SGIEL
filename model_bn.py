import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet18, resnet50, resnet101
from loss import sce, OriTripletLoss, shape_cpmt_cross_modal_ce
# 采用 L2 范数按照第二个维度对输入张量x进行归一化操作
# 与 BatchNorm 等标准归一化层的主要区别在于：
# Normalize 不是通过均值和标准差来归一化输入，而是通过计算每个向量的范数来对其进行单位化
# BatchNorm 是为了稳定网络的训练，减轻梯度问题；
# 而 Normalize 是在输入的尺度不重要时，通过归一化向量使得模型专注于特征间的方向差异。
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        # self.inter_channels = reduc_ratio//reduc_ratio 我觉得这个地方写错了,修改：
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        self.Wbn_shape = nn.BatchNorm2d(self.in_channels)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        nn.init.constant_(self.Wbn_shape.weight, 0.0)
        nn.init.constant_(self.Wbn_shape.bias, 0.0)


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x, shape=False):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''
        # 在图像中x:(b,c,h,w)
        # 在视频序列中x:(b,c,t,h,w)
        batch_size = x.size(0)
        # self.g(x) 的输出形状为 (batch_size, inter_channels, t, h, w)
        # self.g(x).view(batch_size, self.inter_channels, -1)将特征图的空间维度t * h * w展平
        # 即将张量的形状变为(batch_size, inter_channels, t * h * w)。
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # 调整张量维度顺序，使其变为(batch_size, t * h * w, inter_channels)
        g_x = g_x.permute(0, 2, 1)

        # theta_x和phi_x都是通过1×1×1卷积生成的，作用是将输入特征x变换为两个不同的投影空间。
        # 两者都将输入x通过卷积压缩成inter_channels，并展平为一维。
        # theta_x的形状是(batch_size, t * h * w, inter_channels)。
        # phi_x的形状是(batch_size, inter_channels, t * h * w)。
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # orch.matmul(theta_x, phi_x) 执行批量矩阵乘法，其中每个批次的矩阵乘法是分别进行的
        # 得到f，它是一个相似性矩阵，形状为(batch_size, t * h * w, t * h * w)
        # 表示输入张量在不同位置之间的相似性。
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)

        #??为什么用1/N而不是softmax进行过归一化?
        # Softmax适合处理那些希望强调高相似度特征的情况，例如在图像理解和分类中，具有高相似度的特征应该更为重要
        # 1 / N更倾向于均匀地融合所有特征，从而更广泛地捕捉全局上下文信息
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        # 通过矩阵乘法得到y的形状为(batch_size, t * h * w, inter_channels)
        # 通过.permute(0, 2, 1)将维度调整为(batch_size, inter_channels, t * h * w)
        # 然后使用.view将其重新调整为与原始输入特征图相同的形状(batch_size, inter_channels, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # 用1×1×1卷积将通道数从inter_channels恢复到输入通道数in_channels
        # 如果 shape=True，则使用形状模态的批归一化 Wbn_shape替代W[1](batchNorm2d)
        if shape:
            W_y = self.Wbn_shape(self.W[0](y))
        else:
            W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
# 初始化卷积层、线性层和批归一化层的权重,使用了 Kaiming He 初始化
# 什么时候用fan_in，什么时候用fan_out?一般默认用fan_in
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # mode = 'fan_in'表示使用输入的特征数量（fan_in）来缩放权重，这对前向传播有益
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        # 使用输出特征数量来缩放权重，适用于反向传播。对于全连接层，使用fan_out通常可以更好地维持梯度的稳定性
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias:
            init.zeros_(m.bias.data)
    # 1D批归一化层（BatchNorm1d）
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
# 对分类器（全连接层）的权重进行初始化
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


# 针对可见光模态（RGB）的模块，使用 ResNet-50 构建，定义了网络的前几层卷积、BatchNorm 和最大池化操作。用于处理RGB图和可见光模态的形状图的输入
class visible_module(nn.Module):
    def __init__(self, isshape, modalbn):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1, isshape=isshape, onlyshallow=True, modalbn=modalbn)
        print('visible module:', model_v.isshape, model_v.modalbn)

        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x, modal=0):
        x = self.visible.conv1(x)
        if modal == 0: # RGB
            bbn1 = self.visible.bn1
        elif modal == 3: # shape
            bbn1 = self.visible.bn1_shape
        x = bbn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

# 针对红外（IR）模态的模块，结构与 visible_module 类似，用于处理 IR图和红外模态的形状图的输入
class thermal_module(nn.Module):
    def __init__(self, isshape, modalbn):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1,isshape=isshape,onlyshallow=True, modalbn=modalbn)
        print('thermal resnet:', model_t.isshape, model_t.modalbn)

        # avg pooling to global pooling
        self.thermal = model_t


    def forward(self, x, modal=1):
        x = self.thermal.conv1(x)
        if modal == 1: # IR
            bbn1 = self.thermal.bn1
        elif modal == 3: # shape
            bbn1 = self.thermal.bn1_shape
        x = bbn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x



# 基础的 ResNet 模块，多模态的共享特征提取层
class base_resnet(nn.Module):
    def __init__(self, isshape, modalbn):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1, isshape=isshape, modalbn=modalbn)
        print('base resnet:', model_base.isshape, model_base.modalbn)
        # avg pooling to global pooling  #??这个有什么用
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x, modal=0):
        x = self.base.layer1(x, modal)
        x = self.base.layer2(x, modal)
        x = self.base.layer3(x, modal)
        x = self.base.layer4(x, modal)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()
        # 根据isshape和modalbn构建base_resnet/thermal_module/visible_module模块
        self.isshape = True
        self.modalbn = 2
        # thermal_module和visible_module的modalbn为1，base_renet的modalbn=2:
        # 由于每个模块只处理一种数据模态，它们不需要跨模态的批归一化，只需在单一模态内进行归一化
        # base_resnet负责同时处理红外（thermal）和可见光（visible）两种模态的数据。
        # 设置modalbn = 2表示这个模块能够处理两个不同模态的数据，并且在归一化过程中会考虑到这两种模态的特性
        self.thermal_module = thermal_module(self.isshape, 1)
        self.visible_module = visible_module(self.isshape, 1)
        self.base_resnet = base_resnet(self.isshape, self.modalbn)
        
        # TODO init_bn or not
        # 对base_resnet/thermal_module/visible_module的BN层进行初始化
        self.base_resnet.base.init_bn()
        self.thermal_module.thermal.init_bn()
        self.visible_module.visible.init_bn()
        self.non_local = no_local
        # 分别在ResNet的各层插入非局部模块（插入数量即为non_layers对应数量）
        # 并使用self.NL_1, self.NL_2, self.NL_3, 和self.NL_4来存储非局部模块
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                # Non_local(in_channel)，对于Non_local，输入通道数最终等于输出通道数，对于每一个block，输出数量以此为256，512，1024，2048
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

       # ResNet的输出通道数inter_channel:2048，即pool的in_channel
        pool_dim = 2048
        kk = 4
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        # 对bottleneck和classifier进行参数的初始化
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


        if self.isshape:
            self.bottleneck_shape = nn.BatchNorm1d(pool_dim)
            self.bottleneck_shape.bias.requires_grad_(False)  # no shift
            self.classifier_shape = nn.Linear(pool_dim//kk, class_num, bias=False)
            # 定义了两个投影矩阵proj和proj_shape，并将它们存储在 projs 中。
            # 这些矩阵是用于将特征从原始的高维空间映射到较低维度的子空间中
            self.projs = nn.ParameterList([])
            proj = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))
            # proj2 = nn.Parameter(torch.zeros([pool_dim,pool_dim//4*3], dtype=torch.float32, requires_grad=True))
            proj_shape = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))

            nn.init.kaiming_normal_(proj, nonlinearity="linear")        
            nn.init.kaiming_normal_(proj_shape, nonlinearity="linear")        
            self.bottleneck_shape.apply(weights_init_kaiming)
            self.classifier_shape.apply(weights_init_classifier)
            self.projs.append(proj)
            self.projs.append(proj_shape)
        if self.modalbn >= 2:
            self.bottleneck_ir = nn.BatchNorm1d(pool_dim)
            self.bottleneck_ir.bias.requires_grad_(False)  # no shift
            self.classifier_ir = nn.Linear(pool_dim//4, class_num, bias=False)
            self.bottleneck_ir.apply(weights_init_kaiming)
            self.classifier_ir.apply(weights_init_classifier)
        if self.modalbn == 3:
            self.bottleneck_modalx = nn.BatchNorm1d(pool_dim)
            self.bottleneck_modalx.bias.requires_grad_(False)  # no shift
            # self.classifier_ = nn.Linear(pool_dim, class_num, bias=False)
            self.bottleneck_modalx.apply(weights_init_kaiming)
            # self.classifier_rgb.apply(weights_init_classifier)
            self.proj_modalx = nn.Linear(pool_dim, pool_dim//kk, bias=False)
            self.proj_modalx.apply(weights_init_kaiming)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

    def forward(self, x1, x2, x1_shape=None, x2_shape=None, mode=0):
        if mode == 0: # training
            # self.visible_module和self.thermal_module提取可见光和红外图像的特征。
            # 如果形状模态数据存在，则通过self.visible_module和self.thermal_module进一步提取形状特征，并将形状特征拼接在一起。
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            if x1_shape is not None:
                x1_shape = self.visible_module(x1_shape, modal=3)
                x2_shape = self.thermal_module(x2_shape, modal=3)
                x_shape = torch.cat((x1_shape, x2_shape), 0)
        # mode = 1为RGB评估
        # mode = 2为IR评估
        elif mode == 1: # eval rgb
            x = self.visible_module(x1)
        elif mode == 2: # eval ir
            x = self.thermal_module(x2)

        # shared block
        if mode > 0: # eval, only one modality per forward
            x = self.base_resnet(x, modal=mode-1)
        else: # training
            x1 = self.base_resnet(x1, modal=0)
            x2 = self.base_resnet(x2, modal=1)
            # torch.cat将x1和x2在第0维度上“堆叠”在一起，形成一个更大的batch。
            # 使用dim = 0（batch_num)行拼接,可以增加模型处理的数据量
            # 假设x1和x2的形状分别是(b1, c, h, w)和(b2, c, h, w),拼接之后x的形状是(b1+b2,c,h,w)
            # 这里即(b1+b1,c,h,w)
            x = torch.cat((x1, x2), 0)
            
        if mode == 0 and x1_shape is not None: # shape for training
            x_shape = self.base_resnet(x_shape, modal=3)

        # gempooling 介于平均池化和全局池化之间的池化 得到x_pool(b+b,c)
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        p = 3.0
        x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        # 若是训练阶段并且具有形状图的话，则还需要对形状图进行GEM池化
        if mode == 0 and x1_shape is not None:
            b, c, h, w = x_shape.shape
            # x_shape(b,c,h*w)
            x_shape = x_shape.view(b, c, -1)
            p = 3.0
            # torch.mean(x ** p, dim=-1)：在dim = -1维度（即每个通道内的像素点）上求均值
            # 得到形状为(batch_size, channels)的张量。
            # torch.mean()在该维度上求得均值后会返回一个将该维度缩减的张量
            # 得到的张量形状为(batch_size, channels)，代表每个通道的平均值
            x_pool_shape = (torch.mean(x_shape**p, dim=-1) + 1e-12)**(1/p)

        # BNNeck
        if mode == 1:
            feat = self.bottleneck(x_pool)
        elif mode == 2:
            feat = self.bottleneck_ir(x_pool)
        elif mode == 0:
            assert x1.shape[0] == x2.shape[0]
            # x_pool[:x1.shape[0]]x_pool[x1.shape[0]:]和的形状：(b1,c)
            feat1 = self.bottleneck(x_pool[:x1.shape[0]])
            feat2 = self.bottleneck_ir(x_pool[x1.shape[0]:])
            feat = torch.cat((feat1, feat2), 0)
        if mode == 0 and x1_shape is not None:
            feat_shape = self.bottleneck_shape(x_pool_shape)

        # shape-erased feature
        if mode == 0:
            if x1_shape is not None:
                feat_p = torch.mm(feat, self.projs[0])
                # 对投影矩阵self.projs[0]进行L2归一化
                proj_norm = F.normalize(self.projs[0], 2, 0)
                # 这一步是将原始特征feat投影到低维子空间，并再次投影回到原始空间
                # 消除部分方向的特征信息，从而得到形状擦除特征 (b,c),(c,c//kk),(c//kk,c)相乘，即得到(b,c)
                # torch.mm(feat, proj_norm)将特征 feat 投影到与形状相关的子空间中，得到该子空间的表示
                # torch.mm(..., proj_norm.t())将投影到形状相关子空间中的表示，再投影回原始特征空间中。
                # 这相当于在原始特征空间中找到与形状相关的部分。则形状擦除的部分为feat-feat_pnpn
                feat_pnpn = torch.mm(torch.mm(feat, proj_norm), proj_norm.t())
                # feat_shape是通过self.visible_module和self.thermal_module提取的形状特征
                # self.projs[1]是一个独立的投影矩阵，用于将形状特征投影到另一个低维空间，形状变为(b,c//kk)
                feat_shape_p = torch.mm(feat_shape, self.projs[1])
                # 分类操作
                #分别对得到的形状擦拭、全局特征、形状相关特征进行分类
                logit2_rgbir = self.classifier(feat-feat_pnpn)
                logit_rgbir = self.classifier(feat)
                logit_shape = self.classifier_shape(feat_shape_p)

                return {'rgbir':{'bef':x_pool, 'aft':feat, 'logit': logit_rgbir, 'logit2': logit2_rgbir,'zp':feat_p,'other':feat-feat_pnpn},'shape':{'bef':x_pool_shape, 'aft':feat_shape, 'logit':logit_shape,'zp': feat_shape_p} }

            else:
                return x_pool, self.classifier(feat)
        else:

            return self.l2norm(x_pool), self.l2norm(feat)

    # 遍历模型中的所有参数（通过named_parameters()方法），返回参数的名称（k）和参数张量（v）
    # if v.requires_grad: 检查该参数是否需要梯度更新
    # 如果参数不需要计算梯度（requires_grad = False），则不会被加入返回的参数列表
    # 返回模型中某些特定层之外的可训练参数，这样可以在优化时只更新那些被选择的参数
    # 而不对分类器、投影矩阵、批归一化层和Bottleneck层的参数进行更新
    # 为什么过滤这些特定层次？
    def myparameters(self):
        res = []
        for k, v in self.named_parameters():
            if v.requires_grad:
                if 'classifier' in k or 'proj' in k or 'bn' in k or 'bottleneck' in k:
                    continue
                res.append(v)
        return res
