import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO --> DONE!
        self.C = 3
        self.in_channels = self.C
        self.middle1_channels = 8
        self.middle2_channels = 16
        self.out_channels = 32
        self.conv = nn.Conv2d
        self.bn = nn.BatchNorm2d
        self.relu = nn.ReLU

        self.network = nn.Sequential(
            self.conv(self.in_channels, self.middle1_channels, kernel_size=3, stride=1, padding=1),
            self.bn(self.middle1_channels),
            self.relu(),
            self.conv(self.middle1_channels, self.middle1_channels, kernel_size=3, stride=1, padding=1),
            self.bn(self.middle1_channels),
            self.relu(),
            self.conv(self.middle1_channels, self.middle2_channels, kernel_size=5, stride=2, padding=2),
            self.bn(self.middle2_channels),
            self.relu(),
            self.conv(self.middle2_channels, self.middle2_channels, kernel_size=3, stride=1, padding=1),
            self.bn(self.middle2_channels),
            self.relu(),
            self.conv(self.middle2_channels, self.middle2_channels, kernel_size=3, stride=1, padding=1),
            self.bn(self.middle2_channels),
            self.relu(),
            self.conv(self.middle2_channels, self.out_channels, kernel_size=5, stride=2, padding=2),
            self.bn(self.out_channels),
            self.relu(),
            self.conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            self.bn(self.out_channels),
            self.relu(),
            self.conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            self.bn(self.out_channels),
            self.relu(),
            self.conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO --> DONE!
        output = self.network(x.float())

        return output


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO --> DONE!
        self.G = G
        self.in_channels = self.G
        self.middle1_channels = 8
        self.middle2_channels = 16
        self.middle3_channels = 32
        self.out_channels = 1
        self.conv = nn.Conv2d
        self.conv_transp = nn.ConvTranspose2d
        self.relu = nn.ReLU

        self.network0 = nn.Sequential(
            self.conv(self.in_channels, self.middle1_channels, 3, stride=1, padding=1),
            self.relu()
        )
        self.network1 = nn.Sequential(
            self.conv(self.middle1_channels, self.middle2_channels, 3, stride=2, padding=1),
            self.relu()
        )
        self.network2 = nn.Sequential(
            self.conv(self.middle2_channels, self.middle3_channels, 3, stride=2, padding=1),
            self.relu()
        )
        self.network3 = self.conv_transp(self.middle3_channels, self.middle2_channels, 3, stride=2, padding=1, output_padding=1)
        self.network4 = self.conv_transp(self.middle2_channels, self.middle1_channels, 3, stride=2, padding=1, output_padding=1)
        self.network_final = self.conv(self.middle1_channels, self.out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO --> DONE!
        b, g, d, h, w = x.size()

        # Reshaping S to feed into network
        # ----------A)-------------
        x = x.transpose(1, 2).reshape(b*d, g, h, w)
        # ----------B)------------
        # x = x.reshape(b, g, d*h, w)

        c0 = self.network0(x.float())
        c1 = self.network1(c0)
        c2 = self.network2(c1)
        c3 = self.network3(c2)
        c4 = self.network4(c3 + c1)
        output = self.network_final(c4 + c0).squeeze(1)

        return output.view(b, d, h, w)


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B, C, H, W = src_fea.size()
    D = depth_values.size(1)

    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO --> DONE!
        pts_hom = torch.stack((x, y, torch.ones(x.shape[0]))).unsqueeze(0).repeat(B, 1, 1).double()
        # Projection calculation
        # --------------A)---------------------------------------------------------
        depth_values = depth_values.unsqueeze(2).unsqueeze(1).repeat(1, 1, 1, H*W)
        p_ij = torch.matmul(rot, pts_hom).unsqueeze(2).repeat(1, 1, D, 1)*depth_values + trans.unsqueeze(2)

        # Eliminating possible sources of errors (negative depths)
        # idx_neg = p_ij[:, 2:] < 0
        # p_ij[:, 0:1][idx_neg] = 1.0
        # p_ij[:, 1:2][idx_neg] = 1.0
        # p_ij[:, 2:3][idx_neg] = 1.0

        # Refactor projected pixels such that third coordinate is 1
        p_ij = p_ij[:, :2, :, :]/p_ij[:, 2:3, :, :]
        # normalize coordinates
        x_norm = p_ij[:, 0, :, :]/(0.5*(W - 1)) - 1
        y_norm = p_ij[:, 1, :, :]/(0.5*(H - 1)) - 1
        grid = torch.stack((x_norm, y_norm), dim=3)

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO --> DONE!
    warped_src_fea = F.grid_sample(src_fea.double(), grid, align_corners=True)

    return warped_src_fea.view(B, C, D, H, W)


def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO --> DONE!
    B, C, D, H, W = warped_src_fea.size()
    ref_fea_g = ref_fea.view(B, G, math.floor(C/G), 1, H, W)
    warped_src_fea_g = warped_src_fea.view(B, G, math.floor(C/G), D, H, W)
    similarity = (warped_src_fea_g*ref_fea_g).mean(2)

    return similarity


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO --> DONE!
    B, D, H, W = p.size()

    depth = torch.sum(p * depth_values.view(B, D, 1, 1), dim=1)

    return depth


def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO --> DONE!
    loss = 0
    for i in range(len(depth_est)):
        # ----------------A)----------------------------------------------
        loss += F.l1_loss(depth_est[i] * mask[i], depth_gt[i] * mask[i])
        # ----------------B)----------------------------------------------
        # mask_element = torch.tensor(1.)
        # loss += F.l1_loss(depth_est[i][mask[i]==mask_element], depth_gt[i][mask[i]==mask_element])

    return loss
