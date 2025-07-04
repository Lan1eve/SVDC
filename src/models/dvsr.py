# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from thop import profile


class HistLoss(nn.Module):
    def __init__(self):
        super(HistLoss, self).__init__()
        self.name = 'Hist'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

    def dvsr_weight(self, d, img):
        """
        generate full dToF histogram using Eq.1 in paper
        """
        albedo = torch.mean(img, dim=1, keepdim=True)
        r = (albedo / (1e-3 + d ** 2))
        return r

    def dtof_hist_torch(self, d, img, rebin_idx, pitch, mask):
        """
        Convert from predicted depth map into a histogram
        Args:
            d (tensor): predicted depth map with size (n*t, 1, h, w)
            img (tensor): input guidance image with size (n*t, 3, h, w)
            rebin_idx (tensor):
                Compression rebin index (see 'datasets/dtof_simulator.py' for details)
                with size (n*t, 2*self.mpeaks+2, h/s, w/s)
            pitch: size of each patch (iFoV), same as self.scale in main model
            temp_res: temporal resolution of dToF sensor
        """
        d = torch.clamp(d.clone(), min=0.0, max=10.0).to(img.device)
        B, _, H, W = d.shape  ## same resolution as final output
        _, M, _, _ = rebin_idx.shape
        albedo = torch.mean(img, dim=1).unsqueeze(1)
        r = albedo / (1e-3 + d ** 2)

        rebin_idx = torch.repeat_interleave(
            torch.repeat_interleave(rebin_idx, pitch, dim=2), pitch, dim=3
        ).detach()      # [1,M,H,W]
        hist = torch.sum(
            ((torch.floor(d / 0.075) - rebin_idx) >= 0).float(), dim=1
        ).unsqueeze(1)  # [1,1,H,W]

        hist[~mask] = 0

        idx_volume = (torch.arange(1, M + 1).unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(img.device))  # [1,M,1,1]
        hist = ((hist - idx_volume) == 0).float()
        # print(hist.shape)
        hist = torch.sum(
            (hist * r).view(B, M, H // pitch, pitch, W // pitch, pitch), dim=(3, 5)
        )

        return hist

    def get_inp_error(self, cdf, rebin_idx, pred, img, mask, pitch, upsample):
        """
        Get histogram matching error
        Args:
            cdf (tensor): input compressed cumulative distribution functions
                with size (n*t, 2*self.mpeaks+2, h/s, w/s)
            rebin_idx (tensor):
                Compression rebin index (see 'datasets/dtof_simulator.py' for details)
                with size (n*t, 2*self.mpeaks+2, h/s, w/s)
            pred (tensor): predicted depth map with size (n*t, 1, h, w)
            img (tensor): input guidance RGB image
                with size  (n*t, 3, h, w)
            pitch: size of each patch (iFoV), same as self.scale in main model
            temp_res:
        """
        if upsample:
            b, _, h, w = img.shape
            # pred = nn.functional.interpolate(pred, img.shape[-2:], mode='bilinear', align_corners=True)
            rebin_idx = rebin_idx.to(pred.device)
            cdf = cdf.to(pred.device)
            img = img.to(pred.device)
            # B, M, h, w = rebin_idx.shape
        else:
            # print(img.shape)
            b, _, h, w = img.shape
            img = img.to(pred.device)
            cdf = cdf.to(pred.device)
            rebin_idx = rebin_idx.to(pred.device)

        h_dtof, w_dtof = int(h/pitch), int(w/pitch)
        cdf = cdf.contiguous().reshape(b,h_dtof,w_dtof,-1).permute(0,3,1,2)
        rebin_idx = rebin_idx.contiguous().reshape(b,h_dtof,w_dtof,-1).permute(0,3,1,2)

        delta_idx = rebin_idx[:, 1:] - rebin_idx[:, :-1]
        cdf_inp = cdf / (torch.max(cdf, dim=1)[0].unsqueeze(1) + 1e-3)                       # [b,18,30,40]

        hist_pred = self.dtof_hist_torch(pred, img, rebin_idx[:, :-1], pitch, mask)            # [b,17,30,40]

        hist_pred = hist_pred / (torch.sum(hist_pred, dim=1).unsqueeze(1) + 1e-3)
        cdf_pred = torch.cumsum(hist_pred, dim=1).detach()                              # [b,17,30,40]
        inp_error = torch.mean(torch.abs((cdf_inp[:, 1:] - cdf_pred) * delta_idx), dim=1).unsqueeze(1)  # [b,1,30,40]

        # if patch's original cdf max=0, set corresponding inp_error = -1
        inp_error[torch.max(cdf_inp, dim=1)[0].unsqueeze(1) == 0] = -1
        del hist_pred, cdf_pred, cdf_inp, rebin_idx
        # print(inp_error.shape)
        inp_error = torch.repeat_interleave(
            torch.repeat_interleave(inp_error, pitch, dim=2), pitch, dim=3
        ).detach()
        # print(inp_error.shape)
        return inp_error

def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def get_pos_encoding(B, H, W, pitch):
    """
    Positional Encoding to assist alignment vector predictions
    Args:
        B, T, H, W: batch size, number of frames, height and weight of sequence
            (same resolution as final output)
        pitch: size of each patch (iFoV), same as self.scale in main model
    """
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y = y.unsqueeze(0).unsqueeze(1).float()
    x = x.unsqueeze(0).unsqueeze(1).float()
    patch_y = -torch.nn.MaxPool2d(kernel_size=pitch, stride=pitch)(-y)
    patch_x = -torch.nn.MaxPool2d(kernel_size=pitch, stride=pitch)(-x)
    patch_y = torch.repeat_interleave(
        torch.repeat_interleave(patch_y, pitch, dim=2), pitch, dim=3
    )
    patch_x = torch.repeat_interleave(
        torch.repeat_interleave(patch_x, pitch, dim=2), pitch, dim=3
    )
    rel_y = y - patch_y
    rel_x = x - patch_x
    abs_pos = torch.cat((y / H, x / W), dim=1)
    rel_pos = torch.cat((rel_y / pitch, rel_x / pitch), dim=1)
    patch_pos = torch.cat((patch_y / H, patch_x / W), dim=1)
    pos_encoding = torch.cat((abs_pos, rel_pos, patch_pos), dim=1).float()
    return pos_encoding.repeat(B, 1, 1, 1)


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale