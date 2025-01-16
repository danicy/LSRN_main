from functools import partial
import torch
import torch.nn as nn
from numba.core.typing.builtins import Print
from torchgen.native_function_generation import self_to_out_signature
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY

# 深度可分离卷积模块，用于轻量化网络结构
class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


# 基于深度可分离卷积的上采样模块
class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

class CCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCA, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)

        return y

# 定义一个带有步幅的卷积层，用于下采样
class StridedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StridedConv, self).__init__()
        # 深度可分离卷积，用于特征图下采样
        self.conv = DepthWiseConv(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1卷积，用于通道数的变换
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 批量归一化，用于稳定训练过程
        self.bn = nn.BatchNorm2d(out_channels)
        # GELU激活函数，用于引入非线性
        self.gelu = nn.GELU()

    def forward(self, x):
        # 深度可分离卷积操作
        x = self.conv(x)
        # 1x1卷积操作
        x = self.conv1x1(x)
        # 批量归一化
        x = self.bn(x)
        # 激活函数
        x = self.gelu(x)
        return x

# 空间注意力机制
class SA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SA, self).__init__()
        # 假设StridedConv、BSConvU和MSFF都不会改变输入的H和W
        self.strided_conv = StridedConv(in_channels, out_channels)
        self.bsconv_u = BSConvU(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x):
        # 通过不同模块提取特征
        x_strided = self.strided_conv(x)
        x1 = self.gelu(x_strided)
        x_bsconv = self.bsconv_u(x1)

        # 确保sigmoid后的结果与输入x的尺寸相同
        assert x_bsconv.size() == x.size(), "Output of SSA should match the input dimensions"
        weight_map = self.sigmoid(x_bsconv)
        return weight_map

class CSMA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CSMA, self).__init__()
        self.cca = CCA(in_channels, reduction=reduction)
        self.ssa = SA(in_channels, in_channels)

    def forward(self, x):
        # 获取CCA输出
        cca_output = self.cca(x)
        # 获取SSA输出
        ssa_output = self.ssa(x)
        # 计算注意力权重
        combined_weights = cca_output * ssa_output
        return x * combined_weights

class LSA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(LSA, self).__init__()
        # 减少中间特征图的通道数以节省计算资源
        f = num_feat // 4
        num_mixed_convs = 2  # 因为有2个混合卷积

        BSConvS_kwargs = {}
        # 深度可分离卷积
        if conv.__name__ == 'BSConvU':
            self.ds_conv = nn.Sequential(
                conv(f, f, kernel_size=3, padding=0, **BSConvS_kwargs),  # 深度卷积
                conv(f, f, kernel_size=1)  # 点卷积
            )
        else:
            self.ds_conv = nn.Sequential(
                DepthWiseConv(f, f, kernel_size=3, padding=0),  # 深度卷积
                conv(f, f, kernel_size=1)  # 点卷积
            )

        # 混合卷积
        self.mixed_conv = nn.ModuleList([
            conv(f, f, kernel_size=7, dilation=2, padding=6, **BSConvS_kwargs),  # k7d2
            conv(f, f, kernel_size=5, dilation=2, padding=4, **BSConvS_kwargs),  # k3d2
        ])

        # 整合混合卷积输出的 1x1 卷积
        self.conv_merge = conv(f * num_mixed_convs, num_feat, kernel_size=1, padding=0, stride=1)

        # 用于调整 reduced_x 通道数的 1x1 卷积
        self.conv_adjust = conv(f, f * num_mixed_convs, kernel_size=1, padding=0, stride=1)

        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 减少通道数
        reduced_x = self.conv_reduce(x)
        # 深度可分离卷积
        ds_x = self.ds_conv(reduced_x)
        # 混合卷积
        mixed_outputs = [conv(ds_x) for conv in self.mixed_conv]
        mixed_outputs = torch.cat(mixed_outputs, dim=1)
        # 调整 reduced_x 的通道数以匹配 mixed_outputs
        adjusted_reduced_x = self.conv_adjust(reduced_x)
        # 将原始输入的 1x1 卷积结果与混合卷积结果相加
        mixed_outputs = adjusted_reduced_x + mixed_outputs
        # 整合混合卷积输出
        merged_x = self.conv_merge(mixed_outputs)
        # 生成注意力掩码
        mask = self.sigmoid(merged_x)
        # 应用注意力掩码
        return x * mask

# 残差块
class DSRB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, **kwargs):
        super(DSRB, self).__init__()
        self.conv1 = BSConvU(in_channels, out_channels, kernel_size=3, **kwargs)
        self.conv2 = BSConvU(out_channels, out_channels, kernel_size=3, **kwargs)
        self.act = nn.GELU()
        self.lsa = LSA(out_channels, conv)

    def forward(self, x):
        r_c0 = self.act(self.conv1(x) + x)
        r_c1 = self.act(self.conv2(r_c0) + r_c0)
        r_c = self.lsa(r_c1 + r_c0 + x)
        return r_c + x

#ARDB 非对残差蒸馏块
class ARDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(ARDB, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = conv(self.remaining_channels, self.dc, (3,1), padding=(1,0))
        self.c1_r = DSRB(self.remaining_channels, self.rc, conv, **kwargs)

        self.c2_d = conv(self.remaining_channels, self.dc, (1,3), padding=(0,1))
        self.c2_r = DSRB(self.remaining_channels, self.rc, conv, **kwargs)

        self.c3 = BSConvU(self.remaining_channels, self.dc, kernel_size=3, padding=1)
        self.c4 = BSConvU(self.remaining_channels, self.dc, kernel_size=1, padding=0)
        self.act = nn.GELU()
        self.csma = CSMA(self.remaining_channels, out_channels)
        self.conv_1x1_to_64 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv64 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        distilled_c2 = self.act(self.c2_d(r_c1 + self.conv_1x1_to_64(distilled_c1)))
        r_c2 = self.c2_r(r_c1 + self.conv_1x1_to_64(distilled_c1))
        r_c3 = self.act(self.c3(r_c2 + self.conv_1x1_to_64(distilled_c2)))
        r_c3 = self.conv_1x1_to_64(r_c3)
        tensors = [distilled_c1, distilled_c2, r_c3]
        out = torch.cat(tensors, dim=1)
        out = self.conv64(out)
        out_fused = self.csma(out)
        return out_fused + input

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class LSRN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='DepthWiseConv', upsampler='pixelshuffledirect', p=0.25):
        super(LSRN, self).__init__()
        kwargs = {'padding': 1, 'bias': False}
        if conv == 'BSConvS':
            kwargs = {'p': p}

        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            self.conv = nn.Conv2d

        # 使用深度可分离卷积减少参数和计算量
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)
        # 添加一个转换层，将原始输入的通道数从 num_in_ch 转换为 num_feat
        self.conv_to_upsample = nn.Conv2d(num_in_ch, num_feat, kernel_size=1, bias=False)

        # 8个ARDB块
        self.B1 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = ARDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        # 减少通道数以减少参数量
        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, bias=False)
        self.GELU = nn.GELU()

        # 使用深度可分离卷积减少参数和计算量
        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        # 优化上采样模块
        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError("Check the Upsampeler. None or not support yet")

    def forward(self, input):
        # 使用转换层调整原始输入的通道数
        input_for_upsample = self.conv_to_upsample(input)

        # Concatenate the input four times along the channel dimension
        input = torch.cat([input, input, input, input], dim=1)
        # Convolutional feature extraction
        out_fea = self.fea_conv(input)
        # Processing through ESDB blocks
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        # Concatenate the outputs of ESDB blocks
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)

        # Further processing
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        # Final convolution and addition
        out_lr = self.c2(out_B) + out_fea

        # Upsampling using the specified upsampler
        output1 = self.upsampler(out_lr)
        output2 = self.upsampler(input_for_upsample)

        # Sum the outputs
        output = output1 + output2

        return output
