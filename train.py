from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import SYSUData, TestData
from data_manager import *
from eval_metrics import eval_sysu
from model_bn import embed_net
from utils import *
import copy
from loss import OriTripletLoss, TripletLoss_WRT, TripletLoss_ADP, sce, shape_cpmt_cross_modal_ce
# from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import pdb
import wandb
# 使用argparse设置并解析参数
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=3, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--date', default='12.22', help='date of exp')

parser.add_argument('--gradclip', default= 11, type=float,
            metavar='gradclip', help='gradient clip')
parser.add_argument('--gpuversion', default= '3090', type=str, help='3090 or 4090')

path_dict = {}
# path_dict['3090'] = ['/home/share/reid_dataset/SYSU-MM01/', '/home/share/fengjw/SYSU_MM01_SHAPE/']
path_dict['3090'] = ['/home/XXiao/reid_dataset/SYSU-MM01/', '/home/XXiao/reid_dataset/SYSU_MM01_SHAPE/']
path_dict['4090'] = ['/home/jiawei/data/SYSU-MM01/', '/home/jiawei/data/SYSU_MM01_SHAPE/']
# 用parse_args()解析参数
args = parser.parse_args()
# 设置CUDA_VISIBLE_DEVICES 环境变量，用来指定程序运行时使用的 GPU 设备。
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 使用 WandB 初始化实验日志系统，并设置实验名称wandb.run.name，使得实验可以在 WandB 上被准确地追踪和记录。
wandb.init(config=args, project='rgbir-reid2')
args.method = args.method + "_gradclip" + str(args.gradclip) + "_seed" + str(args.seed)
wandb.run.name = args.method
# set_seed(args.seed)
# 设置数据集和日志路径
dataset = args.dataset
if dataset == 'sysu':
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible

# 设置检查点和日志路径
checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)
# 生成实验名称后缀 dataset+_(method)+_p(num_pos)+_n(batch_size)+_lr(lr)
suffix = dataset
# if args.method == 'adp':
#     suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
# else:
suffix = suffix + '_{}'.format(args.method)

suffix = suffix + '_p{}_n{}_lr_{}'.format( args.num_pos, args.batch_size, args.lr)  
# 如果优化器不是 SGD，则在 suffix 中添加优化器的名称
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

# 重定向标准输出到日志文件,将所有的 print() 输出将保存到指定路径的日志文件中。
# 路径格式为 log_path + args.date + '/' + suffix + '_os.txt'，确保每次实验的日志都会被保存并与其他实验区分开来。
sys.stdout = Logger(log_path + args.date + '/' + suffix + '_os.txt')
# 创建可视化日志目录
vis_log_dir = args.vis_log_path + args.date + '/' + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
# writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy 最佳测试精度
best_acc_ema = 0  # best test accuracy
# start_epoch记录训练的起始 epoch，通常用于从中断的地方恢复训练。
start_epoch = 0

print('==> Loading data..')

# Data loading code
# 训练集变换：图像在训练前经过若干数据增强步骤，包括将图像转为 PIL 格式、随机裁剪、随机水平翻转等。
# 测试集变换：只对测试图像进行调整大小和归一化处理，没有数据增强操作
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]
    
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])


transform_train = transforms.Compose( transform_train_list )

# 貌似多余？
end = time.time()
# 根据data_dir读取数据，并进行预处理，调用dataloader的process_query_sysu和process_gallery_sysu去划分测试集数据
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_dir=path_dict[args.gpuversion][0],data_dir1=path_dict[args.gpuversion][1])
    # generate the idx of each person identity
    # 通过GenIdx函数生成训练集中每个身份的索引，用于后续数据采样。
    # GenIdx函数的作用是根据输入的标签数组生成每个身份ID对应的索引列表:
    # 遍历每个标签，将具有相同身份ID的样本索引收集起来。
    # 最终返回两个列表color_pos和thermal_pos，分别表示每个身份ID对应的可见光模态和红外模态的索引位置。
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(mode=args.mode,data_path_ori=path_dict[args.gpuversion][0])
    # gall_img, gall_label, gall_cam = process_gallery_sysu_all(mode=args.mode,data_path_ori=path_dict[args.gpuversion][0])
    gall_img, gall_label, gall_cam = process_gallery_sysu(mode=args.mode, trial=0, data_path_ori=path_dict[args.gpuversion][0]) 

set_seed(args.seed)

# 根据process_query_sysu和process_gallery_sysu处理得到的文件构建gallery query数据集
gallset  = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
# 为gallery set和query set创建数据加载器，使用图像变换处理transform_test，并设置加载时的 batch 大小和 worker 数量。
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

# 获取可见光图像中所有不同身份的标签，计算出数据集中有多少个不同的身份。
# np.unique 是 NumPy 库中的一个函数，它会返回输入数组中所有不同的元素
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..', args.method)

# 模型使用gmplool但不用nonlocal
net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch)
net_ema: embed_net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch)
print('use model without nonlocal but gmpool')
# 将 net和net_ema 移动到指定的计算设备上（通常是 GPU），以确保训练能够在加速硬件上进行。
net.to(device)
net_ema.to(device)
# cudnn.benchmark = True

# 模型恢复与加载
if len(args.resume) > 0:
    # model_path+=args.resume
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        # 在训练模型时，通常会将训练轮次与模型参数一起保存，以便于中断后能够继续训练。
        # checkpoint['epoch']用于获取保存时的轮次编号
        # start_epoch = checkpoint['epoch']则是为了让训练循环从正确的轮次继续。
        start_epoch = checkpoint['epoch']
        # net.load_state_dict(checkpoint['net'])会将这个字典中的权重和偏置值(断点位置）
        # 加载到当前的模型net中，从而恢复训练时的模型状态。
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
# 定义损失函数
criterion_id = nn.CrossEntropyLoss()
if 'agw' in args.method:
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
# 将损失函数criterion_id和criterion_tri移动到指定的计算设备上（通常是GPU），以确保训练能够在加速硬件上进行。
criterion_id.to(device)
criterion_tri.to(device)
# 优化器初始化 参数分组
if args.optim == 'sgd':
    ignored_params = list(map(id, net.classifier.parameters()))
    ignored_params += list(map(id, net.bottleneck.parameters()))
    ignored_params += list(map(id, net.bottleneck_ir.parameters()))
    ignored_params += list(map(id, net.classifier_ir.parameters())) 
    if hasattr(net,'classifier_shape'):
        ignored_params += list(map(id, net.classifier_shape.parameters())) 
        ignored_params += list(map(id, net.bottleneck_shape.parameters())) 
        
        ignored_params += list(map(id, net.projs.parameters())) 
        print('#####larger lr for shape#####')
    # 过滤掉ignored_params后的参数为base_params
    # base_params：预训练参数（通常设置更小的学习率）
    # 在许多深度学习任务中，使用预训练模型作为基础网络可以大大加速收敛，并且有助于提升模型性能。然而，预训练模型的参数不需要太大的调整，因此通常会给这些基础参数设置一个较小的学习率（例如
    # 0.1 * args.lr）。
    # 相反，新加的层，比如分类器或任务相关的层，由于需要从随机初始化开始训练，因此需要一个更高的学习率（通常为args.lr）来加速这些层的训练。
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    params = [{'params': base_params, 'lr': 0.1 * args.lr}, {'params': net.classifier.parameters(), 'lr': args.lr},]
    
    params.append({'params': net.bottleneck.parameters(), 'lr': args.lr})
    params.append({'params': net.bottleneck_shape.parameters(), 'lr': args.lr})
    params.append({'params': net.classifier_ir.parameters(), 'lr': args.lr})
    params.append({'params': net.classifier_shape.parameters(), 'lr': args.lr})
    params.append({'params': net.projs.parameters(), 'lr': args.lr})
    params.append({'params': net.bottleneck_ir.parameters(), 'lr': args.lr})

    # optimizer = optim.Adam(params, weight_decay=5e-4)
    # 定义了一个随机梯度下降（SGD）优化器：
    # params：传入上述分组后的参数，允许每个参数组使用不同的学习率。
    # weight_decay = 5e-4：设置权重衰减（L2正则化）参数，防止过拟合。
    # momentum = 0.9：使用动量加速SGD，减少震荡。
    # nesterov = True：启用Nesterov动量算法，这是一种改进的动量方法，通常比标准动量收敛更快。
    optimizer = optim.SGD(params, weight_decay=5e-4, momentum=0.9, nesterov=True)

# 我们使用 SGD 优化器，随机初始化参数的初始学习率为 0.1，预训练参数的初始学习率为 0.01。
# 对于 SYSU-MM01 和 RegDB 数据集，我们训练模型 100 个 epoch，并在第 20 个和第 50 个 epoch 将学习率降低了 10 倍。
# 我们随机抽取了 8 个身份，每个身份包含 4 个可见光图像和 4 个红外图像来构建一个小批量。
# 对于 HITSZ-VCM 数据集，我们训练了模型 200 个 epoch，并在第 35 个和第 80 个 epoch 将学习率降低了 10 倍。
# 我们随机采样了 8 个身份，每个身份涉及 2 个可见光和 2 个红外轨迹，并为每个轨迹随机采样 3 帧以构建一个小批量。
# 论文描述好像和adjust_learning_rate写的不一致？？

# 前 10 个 epoch 学习率逐渐增加，接下来保持为原始学习率。
# 第 20 到 85 个 epoch 学习率降为原始值的 1/10，最后阶段进一步降为 1/100。
# 这样的策略可以在训练初期加快收敛速度，后期则更稳定地优化模型。
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    ema_w = 1000
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 85:
        lr = args.lr * 0.1
        ema_w = 10000
    elif epoch < 120:
        lr = args.lr * 0.01
        ema_w = 100000
    optimizer.param_groups[0]['lr'] = 0.1*lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr, ema_w

# EMA对模型参数进行历史值的加权平均来平滑更新，从而减少训练中的噪声和波动，使得模型更加稳定，并避免参数剧烈变化
def update_ema_variables(net, net_ema, alpha, global_step=None):
    # torch.no_grad()表示不需要计算梯度，因为这里是手动更新net_ema的参数，而不是通过反向传播来更新
        with torch.no_grad():
            # 通过zip将net和net_ema两个模型的参数逐一配对，循环遍历每个参数的key和value，并分别更新net_ema中的参数
            for ema_item, new_item in zip(net_ema.named_parameters(), net.named_parameters()):
                ema_key, ema_param = ema_item
                new_key, new_param = new_item
                # 对某些特定层（classifier, bottleneck, projs）应用加速的EMA新，使用的alpha变为alpha * 2
                # 这种做法意味着这些层的EMA参数会比其他层更快地接近当前模型的参数。这可能是因为这些层的变化速度较快或者对训练结果影响较大
                # 因此希望它们在EMA中反映出更多的变化。
                if 'classifier' in ema_key or 'bottleneck' in ema_key or 'projs' in ema_key:
                    alpha_now = alpha*2
                else:
                    alpha_now = alpha
                mygrad = new_param.data - ema_param.data
                ema_param.data.add_(mygrad, alpha=alpha_now)

# 函数随机生成一个矩形区域，该矩形区域在图像上被裁剪出来，通常用于数据增强方法
# 通过 cut_rat = np.sqrt(1. - lam) 计算出一个裁剪比例
# cx 和 cy 是在图像的宽度和高度范围内随机生成的中心坐标，用于确定裁剪区域的位置。
# 根据随机生成的中心点 cx 和 cy，计算出裁剪区域的左上角和右下角坐标，即 bbx1, bby1, bbx2, bby2
# np.clip() 将输入值限制在指定范围内，保计算出来的边界不会超出图像的范围
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(epoch):

    current_lr, ema_w = adjust_learning_rate(optimizer, epoch)
    print('current lr', current_lr)
    # AverageMeter是一个辅助类，用于记录和更新各类损失和时间指标的平均值和当前值，便于后续打印和观察训练进度。
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    id_loss_shape = AverageMeter()
    id_loss_shape2 = AverageMeter()
    mutual_loss = AverageMeter()
    mutual_loss2 = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode 切换到训练模式
    net.train()
    net_ema.train()
    end = time.time()
    # trainset经过Dataloader加载后得到trainloader
    # 从数据加载器trainloader中读取一批训练数据，每批包含输入图像x1、x2，以及对应的形状信息x1_shape、x2_shape和标签y1、y2
    for batch_idx, (inputs) in enumerate(trainloader):
        x1, x1_shape, x2, x2_shape, y1, y2 = inputs
        # y1,y2在batchnum维度上经过cat拼接成y
        y = torch.cat((y1, y2), 0)
        # 将x1, x1_shape, x2, x2_shape, y1, y2, y传入到cuda中
        x1, x1_shape, x2, x2_shape, y1, y2, y = x1.cuda(), x1_shape.cuda(), x2.cuda(), x2_shape.cuda(), y1.cuda(), y2.cuda(), y.cuda()
        # data_time更新并记录训练时间
        data_time.update(time.time() - end)

        # 随机得到cutmix处理的概率
        cutmix_prob = np.random.rand(1)
        # 若cutmix_prob概率小于0.2 则进行cutmix操作
        if cutmix_prob < 0.2:
            # generate mixed sample
            #     x1,x2cat拼接得到x,x1_shape,x2_shape cat拼接得到x
            x = torch.cat((x1, x2), 0)
            x_shape = torch.cat((x1_shape, x2_shape), 0)
            # 用Beta分布生成混合比例lam
            lam = np.random.beta(1, 1)
            # 使用rand_bbox()函数生成裁剪区域的坐标
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            # torch.randperm()随机打乱样本顺序,，生成两个目标标签 target_a 和 target_b，对应于原始标签和打乱后的标签,用于后续计算混合损失
            rand_index = torch.randperm(y1.size()[0]).cuda()
            target_a = y
            target_b = torch.cat((y2[rand_index],y1[rand_index]), 0)
            # 对x和x_shape进行cutmix处理，将打乱后的x1,x2和x1_shape和x2_shape进行cut拼接
            x[:, :, bbx1:bbx2, bby1:bby2] = torch.cat((x2[rand_index, :, bbx1:bbx2, bby1:bby2],x1[rand_index, :, bbx1:bbx2, bby1:bby2]),0)
            x_shape[:, :, bbx1:bbx2, bby1:bby2] = torch.cat((x2_shape[rand_index, :, bbx1:bbx2, bby1:bby2],x1_shape[rand_index, :, bbx1:bbx2, bby1:bby2]),0)

            # adjust lambda to exactly match pixel ratio
            # 调整混合比例lambda ，使得其精确反映图像混合后的面积比例。
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

            # compute output
            # 将得到的x和x_shape分成两半然后分别作为新的x1,x2,y1,y2放入net计算output
            outputs = net(x[:y1.shape[0]], x[y1.shape[0]:], x_shape[:y1.shape[0]], x_shape[y1.shape[0]:])
            with torch.no_grad():
                outputs_ema = net_ema(x[:y1.shape[0]], x[y1.shape[0]:], x_shape[:y1.shape[0]], x_shape[y1.shape[0]:])
            # 计算损失（CutMix情况）
            loss_id = criterion_id(outputs['rgbir']['logit'], target_a) * lam + criterion_id(outputs['rgbir']['logit'], target_b) * (1. - lam)
            loss_id2 = torch.tensor([0]).cuda()
            loss_id_shape = criterion_id(outputs['shape']['logit'], target_a) * lam + criterion_id(outputs['shape']['logit'], target_b) * (1. - lam)
            loss_tri = torch.tensor([0]).cuda() 
            loss_kl_rgbir = sce(outputs['rgbir']['logit'][:x1.shape[0]],outputs['rgbir']['logit'][x1.shape[0]:])+sce(outputs['rgbir']['logit'][x1.shape[0]:],outputs['rgbir']['logit'][:x1.shape[0]])
            w1 = torch.tensor([1.]).cuda()
            loss_estimate = torch.tensor([0]).cuda()
            w2 = torch.tensor([1.]).cuda()
            loss_kl_rgbir2 = torch.tensor([0]).cuda()

        # 如果没有触发CutMix
        # 则直接通过net()和net_ema()进行标准的前向传播，并计算各类损失
        # 损失如何计算 先看loss.py
        else:
            with torch.no_grad():
                outputs_ema = net_ema(x1, x2, x1_shape, x2_shape)
            outputs = net(x1, x2, x1_shape, x2_shape)

            # id loss
            # if epoch < 40:
            loss_id = criterion_id(outputs['rgbir']['logit'], y)
            loss_id2 = criterion_id(outputs['rgbir']['logit2'], y)
            loss_id_shape = criterion_id(outputs['shape']['logit'], y)

            # triplet loss
            loss_tri, batch_acc = criterion_tri(outputs['rgbir']['bef'], y)
            
            # cross modal distill
            loss_kl_rgbir = sce(outputs['rgbir']['logit'][:x1.shape[0]],outputs['rgbir']['logit'][x1.shape[0]:])+sce(outputs['rgbir']['logit'][x1.shape[0]:],outputs['rgbir']['logit'][:x1.shape[0]])
            # shape complementary
            loss_kl_rgbir2 = shape_cpmt_cross_modal_ce(x1, y1, outputs)

            # shape consistent        
            loss_estimate =  ((outputs['rgbir']['zp']-outputs_ema['shape']['zp'].detach()) ** 2).mean(1).mean() + sce(torch.mm(outputs['rgbir']['zp'], net.classifier_shape.weight.data.detach().t()), outputs_ema['shape']['logit'])
            

            ############## reweighting ###############
            compliment_grad = torch.autograd.grad(loss_id2+loss_kl_rgbir2, outputs['rgbir']['bef'], retain_graph=True)[0]
            consistent_grad = torch.autograd.grad(loss_estimate, outputs['rgbir']['bef'], retain_graph=True)[0]

            with torch.no_grad():
                compliment_grad_norm = (compliment_grad.norm(p=2,dim=-1)).mean()
                consistent_grad_norm = (consistent_grad.norm(p=2,dim=-1)).mean()
                w1 = consistent_grad_norm / (compliment_grad_norm+consistent_grad_norm) * 2
                w2 = compliment_grad_norm / (compliment_grad_norm+consistent_grad_norm) * 2  

            ############## orthogonalize loss ###############
        proj_inner = torch.mm(F.normalize(net.projs[0], 2, 0).t(), F.normalize(net.projs[0], 2, 0))
        eye_label = torch.eye(net.projs[0].shape[1],device=device)
        loss_ortho = (proj_inner - eye_label).abs().sum(1).mean()


        # 得到总损失loss
        loss = loss_id + loss_tri + loss_id_shape + loss_kl_rgbir + w1*loss_estimate + w2*loss_id2 +w2*loss_kl_rgbir2 + loss_ortho

        if not check_loss(loss):
            import pdb
            pdb.set_trace()
        # 用当前批次数据做训练时，应当先将优化器的梯度置零：
        optimizer.zero_grad()
        # 将loss反向传播回网络：
        loss.backward()
        # 为了防止梯度爆炸，使用clip_grad_norm_()进行梯度裁剪。
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradclip)
        # 使用优化器更新模型参数：
        optimizer.step()

        # EMA参数更新
        update_ema_variables(net, net_ema, 1/ema_w)


        # update P 计算损失并更新
        train_loss.update(loss_id2.item(), 2 * x1.size(0))
        id_loss.update(loss_id.item(), 2 * x1.size(0))
        id_loss_shape.update(loss_id_shape.item(), 2 * x1.size(0))
        id_loss_shape2.update(w1.item(), 2 * x1.size(0))
        mutual_loss2.update(loss_kl_rgbir2.item(), 2 * x1.size(0))
        mutual_loss.update(loss_ortho.item(), 2 * x1.size(0))
        # kl_loss.update(loss_kl2.item()+loss_kl.item() , 2 * x1.size(0))
        kl_loss.update(loss_estimate.item(), 2 * x1.size(0))
        total += y.size(0)

        # measure elapsed time   100. * correct / total 测试时长
        batch_time.update(time.time() - end)
        end = time.time()
        # batch每50次打印一次训练进度
        if batch_idx % 50 == 0:
            # import pdb
            # pdb.set_trace()
            print('Epoch:[{}][{}/{}]'
                  'L:{id_loss.val:.4f}({id_loss.avg:.4f}) '
                  'L2:{train_loss.val:.4f}({train_loss.avg:.4f}) '
                  'sL:{id_loss_shape.val:.4f}({id_loss_shape.avg:.4f}) '
                  'w1:{id_loss_shape2.val:.4f}({id_loss_shape2.avg:.4f}) '
                  'or:{mutual_loss.val:.4f}({mutual_loss.avg:.4f}) '
                  'ML2:{mutual_loss2.val:.4f}({mutual_loss2.avg:.4f}) '
                  'KL:{kl_loss.val:.4f}({kl_loss.avg:.4f}) '.format(
                epoch, batch_idx, len(trainloader),
                train_loss=train_loss, id_loss=id_loss, id_loss_shape=id_loss_shape, id_loss_shape2=id_loss_shape2, mutual_loss=mutual_loss, mutual_loss2=mutual_loss2, kl_loss=kl_loss))


def test(net):
    pool_dim = 2048
    # 对这些图像进行水平翻转
    def fliplr(img):
        '''flip horizontal'''
        # img.size(3)：获取图像的宽度W，即张量的第四维度的大小。
        # torch.arange(img.size(3) - 1, -1, -1)：生成一个从W - 1到0递减的索引序列。
        # 这表示从图像的最后一个列开始到第一个列，从而实现水平翻转
        # .long()：将生成的索引序列转为long类型张量，确保它可以被index_select函数使用
        inv_idx = torch.arange(img.size(3)-1,-1,-1,device=img.device).long()  # N x C x H x W
        # 用于在指定维度上根据给定的索引顺序选择张量中的元素。
        img_flip = img.index_select(3,inv_idx)
        return img_flip
    def extract_gall_feat(gall_loader):
        net.eval()
        print ('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        ngall = len(gall_loader.dataset)
        # 初始化gall_feat_pool和gall_feat_fc
        # feat_pool 和 feat_fc：是从网络中提取的特征，分别代表池化层输出和全连接层输出。
        gall_feat_pool = np.zeros((ngall, pool_dim))
        gall_feat_fc = np.zeros((ngall, pool_dim))
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(gall_loader):
                batch_num = input.size(0)
                # 原始图像和水平翻转图像的特征进行平均，以增强特征的鲁棒性
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, mode=test_mode[0])
                input2 = fliplr(input)
                feat_pool2, feat_fc2 = net(input2, input2, mode=test_mode[0])
                feat_pool = (feat_pool+feat_pool2)/2
                feat_fc = (feat_fc+feat_fc2)/2
                # 将GPU张量转移到CPU并转换为NumPy数组，一次转换一批次
                # ptr是当前处理图像的批次在总特征数组中的起始位置，ptr + batch_num是这个批次结束的位置。
                # 通过这种方式，确保每个批次的特征都存储在gall_feat_pool和gall_feat_fc的正确位置
                gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
                gall_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        # 记录时长
        print('Extracting Time:\t {:.3f}'.format(time.time()-start))
        return gall_feat_pool, gall_feat_fc

    # 和gallery的提取一样，唯一的不同在于net测试时选择的mode不同 mode=1为可见光 mode=2为红外
    def extract_query_feat(query_loader):
        net.eval()
        print ('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        nquery = len(query_loader.dataset)
        query_feat_pool = np.zeros((nquery, pool_dim))
        query_feat_fc = np.zeros((nquery, pool_dim))
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, mode=test_mode[1])
                input2 = fliplr(input)
                feat_pool2, feat_fc2 = net(input2, input2, mode=test_mode[1])
                feat_pool = (feat_pool+feat_pool2)/2
                feat_fc = (feat_fc+feat_fc2)/2
                query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
                query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num         
        print('Extracting Time:\t {:.3f}'.format(time.time()-start))
        return query_feat_pool, query_feat_fc
    # switch to evaluation mode
    net.eval()
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)

    # gall_img, gall_label, gall_cam = process_gallery_sysu(mode=args.mode, trial=0)

    trial_gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
    # ??为什么要这么写？而不是gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader)不同的是num_workers不同，默认为8，这里是4
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

    # pool5 feature
    # 计算池化特征之间的距离矩阵
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    # 调用 eval_sysu 函数，计算 CMC 曲线和 mAP 等评估指标
    cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

    # fc feature
    # 算全连接层特征之间的距离矩阵
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    all_cmc = cmc
    all_mAP = mAP
    all_mINP = mINP
    all_cmc_pool = cmc_pool
    all_mAP_pool = mAP_pool
    all_mINP_pool = mINP_pool
    return all_cmc, all_mAP 



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# training
print('==> Start Training...')
# 总共 120 个 epoch，这个循环中每一轮（epoch）模型会进行训练和评估。
for epoch in range(start_epoch, 120 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    # print(trainset.cInde
    # print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos
    # DataLoader用于加载训练数据
    # loader_batch表示每次批量训练的总样本数。
    # sampler指定样本的选择方式
    # drop_last = True表示如果最后一批次的样本数不足，将丢弃它
    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    # 在第一个epoch，初始化EMA模型net_ema，它的初始权重与原模型net的权重相同。
    if epoch == 0:
        net_ema.load_state_dict(net.state_dict())
        print('init ema modal')

    # 调用train执行本轮的训练
    train(epoch)

    print('Test Epoch: {}'.format(epoch))

    # testing
    # 调用test执行本轮的测试，得到cmc和mAP指标
    # 使用wandb.log()记录rank1 mAP rank1_ema mAP_ema
    cmc, mAP = test(net)
    wandb.log({'rank1': cmc[0],
                'mAP': mAP,
                },step=epoch)
    cmc_ema, mAP_ema = test(net_ema)
    wandb.log({'rank1_ema': cmc_ema[0],
                'mAP_ema': mAP_ema,
                },step=epoch)
    # save model
    # 保存最优结果的net模型和net_ema模型
    if cmc[0] > best_acc: 
        best_acc = cmc[0]
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'cmc': cmc,
            'mAP': mAP,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_best.t')
    if cmc_ema[0] > best_acc_ema:  
        best_acc_ema = cmc_ema[0]
        best_epoch_ema = epoch
        state = {
            'net': net_ema.state_dict(),
            'cmc': cmc_ema,
            'mAP': mAP_ema,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_ema_best.t')

    # 每5个epoch定期保存一次EMA模型，以便可以在后续检查模型在不同训练阶段的状态。
    if epoch % 5 == 0:
        state = {
            'net': net_ema.state_dict(),
        }
        torch.save(state, checkpoint_path + suffix + '_' + str(epoch) + '_.t')

    print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print('Best Epoch [{}]'.format(best_epoch))
      
    print('------------------ema eval------------------')
    print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc_ema[0], cmc_ema[4], cmc_ema[9], cmc_ema[19], mAP_ema))
    print('Best Epoch [{}]'.format(best_epoch_ema))
