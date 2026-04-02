
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class KpLoss(object):
    def __init__(self):
        # 实例化了一个均方损失误差适实例
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])

        ########## nrx test 绘制热图 ##########
        # batch_index = 0
        # channel_index = 0
        #
        # # 提取指定批次和通道的热图数据
        # heatmap = heatmaps.clone().cpu()[batch_index, channel_index, :, :]
        # # heatmap_numpy = heatmap.numpy()
        # # 绘制热图
        # # 清除当前的图表，以确保不在旧图上绘制
        # plt.clf()
        # cax = plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        # plt.colorbar(cax, fraction=0.046, pad=0.04)  # 显示颜色条
        # # heatmap_save_path = output_dir+"\\"+img_name+"heatmap.png"
        # # plt.savefig(heatmap_save_path, format='png', dpi=500)
        # plt.show()
        #####################################

        # [num_kps] -> [B, num_kps]
        # kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        # 乘 2 放大损失值
        loss = torch.sum(loss*2)/bs
        i=1
        # loss = torch.sum(loss * kps_weights) / bs
        return loss


class FocalMSELoss(object):
    def __init__(self, gamma=2.0, alpha=0.5):
        """
        对比度感知的 Focal MSE 损失
        :param gamma: Focal Loss 变换参数，增强低置信度区域的学习
        :param alpha: 对比度加权系数
        """
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits, targets):
        """
        计算 Focal MSE Loss，并结合对比度信息进行加权
        :param logits: 预测热力图 [B, num_kps, H, W]
        :param targets: 目标数据（包含 heatmap 和对比度信息）
        :return: 计算出的损失值
        """
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]

        # 目标热力图，仅基于高斯核生成 [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])

        # 计算逐像素 MSE
        mse_loss = self.criterion(logits, heatmaps)

        # 计算 Focal Loss 权重：增强低置信度区域的学习
        focal_weight = (1 - heatmaps) ** self.gamma

        # 结合两种权重
        weighted_loss =  focal_weight * mse_loss

        # 计算 batch-wise 损失，匹配 `KpLoss` 的计算逻辑
        loss = weighted_loss.mean(dim=[2, 3])  # 平均 H, W 维度
        loss = torch.sum(loss * 2) / bs  # 乘 2 放大损失值

        return loss

class ContrastonllyFocalMSELoss(object):
    def __init__(self, gamma=2.0, alpha=0.5):
        """
        对比度感知的 Focal MSE 损失
        :param gamma: Focal Loss 变换参数，增强低置信度区域的学习
        :param alpha: 对比度加权系数
        """
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits, targets):
        """
        计算 Focal MSE Loss，并结合对比度信息进行加权
        :param logits: 预测热力图 [B, num_kps, H, W]
        :param targets: 目标数据（包含 heatmap 和对比度信息）
        :return: 计算出的损失值
        """
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]

        # 目标热力图，仅基于高斯核生成 [B, num_kps, H, W]
        heatmaps = torch.stack([t["contrast_map"].to(device) for t in targets])

        # 计算逐像素 MSE
        mse_loss = self.criterion(logits, heatmaps)

        # 计算 Focal Loss 权重：增强低置信度区域的学习
        focal_weight = (1 - heatmaps) ** self.gamma

        # 结合两种权重
        weighted_loss = focal_weight * mse_loss

        # 计算 batch-wise 损失，匹配 `KpLoss` 的计算逻辑
        loss = weighted_loss.mean(dim=[2, 3])  # 平均 H, W 维度
        loss = torch.sum(loss * 2) / bs  # 乘 2 放大损失值

        return loss


class ContrastFocalMSELoss(object):
    def __init__(self, gamma=2.0, alpha=0.5):
        """
        对比度感知的 Focal MSE 损失
        :param gamma: Focal Loss 变换参数，增强低置信度区域的学习
        :param alpha: 对比度加权系数
        """
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits, targets):
        """
        计算 Focal MSE Loss，并结合对比度信息进行加权
        :param logits: 预测热力图 [B, num_kps, H, W]
        :param targets: 目标数据（包含 heatmap 和对比度信息）
        :return: 计算出的损失值
        """
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]

        # 目标热力图，仅基于高斯核生成 [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])

        # 对比度信息 [B, num_kps, H, W]，用于动态加权
        contrast_maps = torch.stack([t["contrast_map"].to(device) for t in targets])

        # 计算逐像素 MSE
        mse_loss = self.criterion(logits, heatmaps)

        # 计算 Focal Loss 权重：增强低置信度区域的学习
        focal_weight = (1 - heatmaps) ** self.gamma

        # 计算对比度加权：高对比度区域权重更大
        contrast_weight = 1 + self.alpha * contrast_maps

        # 结合两种权重
        weighted_loss = contrast_weight * focal_weight * mse_loss

        # 计算 batch-wise 损失，匹配 `KpLoss` 的计算逻辑
        loss = weighted_loss.mean(dim=[2, 3])  # 平均 H, W 维度
        loss = torch.sum(loss * 2) / bs  # 乘 2 放大损失值

        return loss


class ContrastFocalMSELoss2(object):
    def __init__(self, gamma=2.0, alpha=0.5):
        """
        对比度感知的 Focal MSE 损失
        :param gamma: Focal Loss 变换参数，增强低置信度区域的学习
        :param alpha: 对比度加权系数
        """
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits, targets):
        """
        计算 Focal MSE Loss，并结合对比度信息进行加权
        :param logits: 预测热力图 [B, num_kps, H, W]
        :param targets: 目标数据（包含 heatmap 和对比度信息）
        :return: 计算出的损失值
        """
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]

        # 目标热力图，仅基于高斯核生成 [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap_gauss"].to(device) for t in targets])

        # 对比度信息 [B, num_kps, H, W]，用于动态加权
        contrast_maps = torch.stack([t["contrast_map"].to(device) for t in targets])

        # 计算逐像素 MSE
        mse_loss = self.criterion(logits, heatmaps)

        # 计算 Focal Loss 权重：增强低置信度区域的学习
        focal_weight = (1 - heatmaps) ** self.gamma

        # 计算对比度加权：高对比度区域权重更大
        contrast_weight = 1 + self.alpha * contrast_maps

        # 结合两种权重
        weighted_loss = contrast_weight * focal_weight * mse_loss

        # 计算 batch-wise 损失，匹配 `KpLoss` 的计算逻辑
        loss = weighted_loss.mean(dim=[2, 3])  # 平均 H, W 维度
        loss = torch.sum(loss * 2) / bs  # 乘 2 放大损失值

        return loss
