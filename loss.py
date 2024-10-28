from turtle import position
from urllib.parse import quote_plus
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size=None, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        # n= batch_size
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        # 计算成对的欧氏距离
        # 与dist_mat=pdist_torch(inputs, inputs)相同
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist.t() 是 dist 的转置矩阵，经过这一操作后，dist[i][j] 中每一项将会是 inputs[i] 和 inputs[j] 特征平方和的加和，
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # addmm_ 是 PyTorch 中的一个矩阵乘法函数，它执行 dist = beta * dist + alpha * inputs @ inputs.t()
        # inputs @ inputs.t() 表示 inputs 和它的转置矩阵 inputs.t() 的矩阵乘法，结果是一个大小为 (batch_size, batch_size) 的矩阵，表示输入样本两两之间的点积。
        # beta = 1, alpha = -2: 这里的计算目的是通过公式简化欧氏距离：dist(xi,xj)=（xi-xj)^2=xi^2+xj^2-2*xi*xj
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)

        # clamp(min=1e-12): 对距离进行裁剪，以避免距离过小（特别是数值误差引发的负数）。
        # sqrt(): 对结果进行开方操作，完成欧氏距离的计算。
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        # mask 是一个二值矩阵，大小同样是 (batch_size, batch_size)。mask[i][j] 表示第 i 个样本和第 j 个样本是否属于同一类。
        # targets 是大小为 (n) 的标签张量，n 是 batch size。每个样本有一个类别标签。
        # targets.expand(n, n) 扩展为 (n, n) 大小矩阵，行和列的每个元素都是样本标签。
        # eq(targets.expand(n, n).t())：比较两个扩展的标签矩阵，返回一个布尔矩阵 mask，表示每对样本是否属于同一类别。
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # dist_ap 中存储该样本与所有正样本对中的最大距离，即最难区分的正样本。
        # dist_an 中存储该样本与所有负样本对中的最小距离，即最难区分的负样本。
        # max()/min()用于找出最大/最小距离，并通过 unsqueeze(0) 增加一个维度，得到该批次样本的最大/最小样本距离array
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        # 期望x1 > x2时(dist_an>dist_ap)即排序为顺序时，传入y = 1  反之 传入y=-1
        # 计算三元组损失
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        # 计算准确率
        # orch.ge(dist_an, dist_ap)：这是 torch.ge() 函数，ge 代表 greater than or equal（大于等于）。
        # 它会比较 dist_an 和 dist_ap，并返回一个与 dist_an 和 dist_ap 相同形状的布尔张量。
        # 如果 dist_an[i] >= dist_ap[i]，对应的位置将为 True，否则为 False
        # 目标是 负样本对的距离大于或等于正样本对的距离
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct    
        
        # modal = (torch.arange(n) // (n/2)).cuda()
        # modalmask = modal.expand(n, n).ne(modal.expand(n, n).t())

# Adaptive weights
def softmax_weights(dist, mask):
    # dist * mask 会将 dist 矩阵中与 mask 对应为 0 的位置变为 0，只保留所需要样本对的距离。
    # torch.max(..., dim=1) 会在每一行中找到最大的距离值，并保持维度不变 (keepdim=True)，这样得到的 max_v 的形状是 (N, 1)。
    # 这样可以避免后续 exp 计算时出现数值溢出问题。
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    # 注意：
    # dist 的形状是 (N, N)，表示一个对称矩阵，包含了每对样本之间的距离。
    # max_v 的形状是 (N, 1)，它表示每一行中的最大值，是一个列向量，形如：
    # max_v = [[max_v1],
    #          [max_v2],
    #          [max_v3],
    #          ...
    #          [max_vN]]
    # 当 dist 和 max_v 进行减法运算时，PyTorch 会自动应用 广播机制。广播机制遵循以下规则：
    # 如果两个张量的维度不同，它们会自动扩展为相同的形状，前提是某个维度的大小是 1 时，可以扩展为匹配另一个张量的维度。
    # 扩展后的 max_v 可以视作：
    # max_v_broadcasted = [[max_v1, max_v1, max_v1, ..., max_v1],
    #                      [max_v2, max_v2, max_v2, ..., max_v2],
    #                      [max_v3, max_v3, max_v3, ..., max_v3],
    #                      ...
    #                      [max_vN, max_vN, max_vN, ..., max_vN]]
    # 因此，diff 矩阵表示距离矩阵减去该行的最大值。形状依然是 (N, N)
    diff = dist - max_v
    # 计算归一化的常数 Z
    # 对diff 中的每个元素计算指数（exp 函数）diff 中的元素已经经过减去每行的最大值，数值稳定性会更好，这减少了数值溢出的可能性
    # torch.exp(diff) * mask 表示我们只对相关的样本（由 mask 指定）进行指数计算和操作，忽略掉不相关的样本
    # dim=1 表示对每行进行求和，得到一个 (N, 1) 的列向量，表示每一行中所有相关元素的指数和。
    # keepdim=True 保持输出的维度，保证结果仍然是 (N, 1)，而不是 (N,)，这样可以与后续操作的广播机制保持一致
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    # Z 的形状是 (N, 1)，通过广播机制，Z 将被扩展为 (N, N)，从而对每行的元素进行归一化
    #  W的每一行元素代表相关样本的 softmax 权重，它们被归一化为总和为 1
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    #  torch.norm(x, 2, axis, keepdim=True) 计算 x 在指定的 axis 上的 L2 范数。
    #  keepdim=True 保持维度一致性，即保留被规约（归一化）的维度，返回一个形状与 x 相同的张量，但在归一化的维度上只包含一个数值。
    #  expand_as(x) 是为了扩展 norm，使其与输入张量 x 的形状相同，这样可以进行逐元素相除。
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

# 加权正负样本距离的三元组损失
class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        # pdist_torch：计算输入 inputs 之间的欧氏距离矩阵shape为(n,n)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N] is_pos 和 is_neg分别表示正样本和负样本的掩码矩阵
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        # dist_ap和dist_an 的shape应该是(n,n)才对??
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        # 计算加权正样本距离的 softmax 权重，is_pos 作为掩码矩阵用于忽略无关的样本。通过 softmax，权重分配给每个正样本对的距离，距离越大的正样本对权重越大。
        # 根据exp函数的特性 对于减去了max_v得到的diff 距离越大的正样本值x 对应x'就更大 则exp(x')对应的权重就更大
        weights_ap = softmax_weights(dist_ap, is_pos)
        # 注意：对负样本对距离取负数是为了将最小的负样本对距离赋予更大的权重（软加权机制倾向于赋予最困难的负样本更大的权重）
        weights_an = softmax_weights(-dist_an, is_neg)
        # ist_ap * weights_ap：按权重加权每个正样本对的距离
        # torch.sum(..., dim=1)：对每一行的正样本对距离进行加权求和，得到每个 anchor 样本与所有正样本的加权距离之和。
        # 最终结果 furthest_positive 的形状为 (N,)，表示每个 anchor 样本与最远（加权求和）的正样本之间的距离。
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        # 同样的，对每一行的负样本对距离进行加权求和，得到每个 anchor 样本与所有负样本的加权距离之和。
        # 最终结果 closest_negative 的形状为 (N,)，表示每个 anchor 样本与最近（加权求和）的负样本之间的距离
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        # furthest_positive.new().resize_as_(furthest_positive)：创建一个与 furthest_positive 大小相同的新张量。
        # .fill_(1)：将该张量的所有元素填充为 1。
        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        # y=1 期望closest_negative - furthest_positive尽可能大于0
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        # 计算准确率
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)
        
        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            # 与TripletLoss_WRT的不同处理就在square上
            # 平方的好处是好处是，它会放大正负样本之间的距离差。
            # 假设正样本和负样本之间的差距较大，那么平方差会迅速增大，从而增加损失值，迫使模型更快地优化这一对样本。
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            # 用 torch.clamp_max 限制了平方差的最大值，防止其变得过大（最大值设为 10）。
            # 这是为了防止某些样本对的差异过大，导致极端的梯度更新，进而影响训练的稳定性。
            diff_pow =torch.clamp_max(diff_pow, max=10)

            # Compute ranking hinge loss
            # y1 = (furthest_positive > closest_negative).float()：如果最远的正样本距离大于最近的负样本距离，则 y1 为 1，否则为 0。
            # y = -(y1 + y2)：由于 y1 和 y2 的值分别为 1 和 0(furthest_positive > closest_negative)，或者 0 和 -1.
            # 最终得到的 y 只有两种可能性，分别是 -1 (furthest_positive > closest_negative)或 1。这构造了一个二分类标签，用来指示正样本和负样本之间的相对距离。
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            # 当y=1(furthest_positive<closest_negative)期望diff_pow尽可能大 鼓励增大差距，使得正负样本之间的差距更大
            # y=-1 期望diff_pow尽可能小 希望通过最小化损失来纠正这种情况
            loss = self.ranking_loss(diff_pow, y)
        
        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct



def sce(new_logits, old_logits):
    # 通过 old_logits 的 softmax 概率乘以 new_logits 的 log-softmax 概率，计算了 softmax cross-entropy。
    # 计算了新的 logit 相对于旧的 logit 的 Kullback-Leibler (KL) 散度
    # 两个张量相乘时，是逐元素相乘的（element-wise multiplication），也就是对应位置的元素进行乘法运算 在这里softmax之后均为(batch_num,channel)
    # mean(0)在0维 即batch_num，进行平均操作（即对每列进行求平均），返回一个新的向量，它的形状是 (channel,)。
    loss_ke_ce = (- F.softmax(old_logits, dim=1).detach() * F.log_softmax(new_logits,dim=1)).mean(0).sum()
    return loss_ke_ce




# 通过形状特征的比较和交叉模态的对比学习计算交叉熵损失。
def shape_cpmt_cross_modal_ce(x1,y1,outputs):
    
    with torch.no_grad():
        batch_size = y1.shape[0]
        rgb_shape_normed = F.normalize(outputs['shape']['zp'][:x1.shape[0]], p=2, dim=1)
        ir_shape_normed = F.normalize(outputs['shape']['zp'][x1.shape[0]:], p=2, dim=1)
        rgb_ir_shape_cossim = torch.mm(rgb_shape_normed,ir_shape_normed.t())
        mask = y1.expand(batch_size,batch_size).eq(y1.expand(batch_size, batch_size).t())
        target4rgb, target4ir = [], []
        # idx_temp = torch.arange(batch_size)
        idx_temp = torch.arange(batch_size,device=rgb_shape_normed.device)
        for i in range(batch_size):
            sorted_idx_rgb = rgb_ir_shape_cossim[i][mask[i]].sort(descending=False)[1]
            sorted_idx_ir = rgb_ir_shape_cossim.t()[i][mask.t()[i]].sort(descending=False)[1]
            target4rgb.append(idx_temp[mask[i]][sorted_idx_rgb[0]].unsqueeze(0))
            target4ir.append(idx_temp[mask.t()[i]][sorted_idx_ir[0]].unsqueeze(0))
        target4rgb = torch.cat(target4rgb)
        target4ir = torch.cat(target4ir)
    loss_top1 = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:][target4rgb]) + sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]][target4ir])
    

    loss_random = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:])+sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]])

    loss_kl_rgbir2 = loss_random+loss_top1

    return loss_kl_rgbir2



def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx