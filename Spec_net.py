import torch
from compressai.layers import GDN
from torch import nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers

#光谱特征提取
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(24, 48, kernel_size=7, stride=1, padding=3)
        # self.conv2 = nn.Conv2d(48, 96, kernel_size=7, stride=1, padding=3)
        # self.conv3 = nn.Conv2d(96, 48, kernel_size=7, stride=1, padding=3)
        # self.conv4 = nn.Conv2d(48, 24, kernel_size=7, stride=1, padding=3)
        # self.conv5 = nn.Conv2d(24, 12, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, i, feat):
        x = torch.cat((i, feat), 1)
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.relu3(self.conv3(x2))
        x4 = self.relu4(self.conv4(x3))
        x5 = self.conv5(x4)
        return x5

class update(nn.Module):
    def __init__(self):
        super(update, self).__init__()
        self.conv = ConvNet()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, feat_course, i):
        feat = self.up(feat_course)
        feat_fine = self.conv(i, feat) + feat
        return feat_fine

#光谱估计网络
class Est_spec_net (nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(2, 2, padding=0)
        self.pool2 = nn.AvgPool2d(2, 2, padding=0)
        self.pool3 = nn.AvgPool2d(2, 2, padding=0)
        self.pool4 = nn.AvgPool2d(2, 2, padding=0)
        self.convnet = ConvNet()
        self.update = update()

    def forward(self, i1):
        batch = i1.size(0)
        h = i1.size(2)
        w = i1.size(3)
        c = i1.size(1)
        feat_zero = torch.zeros([batch, c, h//32, w//32]).to(i1.device)
        i_4 = i1
        i_3 = self.pool1(i_4)
        i_2 = self.pool2(i_3)
        i_1 = self.pool3(i_2)
        i_0 = self.pool4(i_1)
        feat_0 = self.update(feat_zero, i_0)
        feat_1 = self.update(feat_0, i_1)
        feat_2 = self.update(feat_1, i_2)
        feat_3 = self.update(feat_2, i_3)
        feat_4 = self.update(feat_3, i_4)
        return feat_4

class Est_spec_net2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(2, 2, padding=0)
        self.pool2 = nn.AvgPool2d(2, 2, padding=0)
        self.pool3 = nn.AvgPool2d(2, 2, padding=0)
        self.pool4 = nn.AvgPool2d(2, 2, padding=0)
        self.convnet = ConvNet()
        self.update = update()

    def forward(self, i1):
        batch = i1.size(0)
        h = i1.size(2)
        w = i1.size(3)
        c = i1.size(1)
        feat_zero = torch.zeros([batch, c, h//16, w//16]).cuda()
        i_4 = i1
        i_3 = self.pool1(i_4)
        i_2 = self.pool2(i_3)
        i_1 = self.pool3(i_2)

        feat_1 = self.update(feat_zero, i_1)
        feat_2 = self.update(feat_1, i_2)
        feat_3 = self.update(feat_2, i_3)
        feat_4 = self.update(feat_3, i_4)
        return feat_4


class ResidualBlock(nn.Module):
    def __init__(self, N):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(N, N, 3, 1, 1)  # 卷积核3，步长1，填充1
        self.conv2 = nn.Conv2d(N, N, 3, 1, 1)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.prelu(x1)
        x3 = self.conv2(x2)
        x4 = self.prelu(x3)
        x5 = torch.add(x, x4)
        return x5

#光谱信息编码器
class SpecEncoder2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.specEncoder = nn.Sequential(
            nn.Conv2d(in_channel, 64, 5, stride=2, padding=2),
            GDN(64),
            ResidualBlock(64),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            GDN(64),
            ResidualBlock(64),
            nn.Conv2d(64, out_channel, 5, stride=2, padding=2),
        )

    def forward(self, spec):
        spec_y = self.specEncoder(spec)
        return spec_y


#光谱信息编码器
class SpecEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpecEncoder, self).__init__()

        self.specEncoder = nn.Sequential(
            nn.Conv2d(in_channel, 64, 5, stride=2, padding=2),
            GDN(64),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            GDN(64),
            nn.Conv2d(64, out_channel, 5, stride=2, padding=2),
        )

    def forward(self, spec):
        spec_y = self.specEncoder(spec)
        return spec_y

#光谱信息解码器
class SpecDecoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.specDecoder = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 64, 5, stride=2, padding=2, output_padding=1),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(64, out_channel, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, spec_y_round):
        spec_hat = self.specDecoder(spec_y_round)
        return spec_hat


class SpecDecoder2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(in_channel, 64, 5, stride=2, padding=2, output_padding=1)
        self.gdn1 = GDN(64, inverse=True)
        self.conv2 = nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1)
        self.gdn2 = GDN(64, inverse=True)
        self.conv3 = nn.ConvTranspose2d(64, out_channel, 5, stride=2, padding=2, output_padding=1)

    def forward(self, spec_y_round):
        spec_hat1 = self.gdn1(self.conv1(spec_y_round))
        spec_hat2 = self.gdn2(self.conv2(spec_hat1))
        spec_hat3 = self.conv3(spec_hat2)
        return spec_y_round, spec_hat1, spec_hat2, spec_hat3


class HyperEncoder(nn.Sequential):
    def __init__(self, c_in, c_mid):
        super().__init__()
        self.HyperEncoder = nn.Sequential(
            conv(c_in, c_mid, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(c_mid, c_mid),
            nn.ReLU(inplace=True),
            conv(c_mid, c_mid),
        )

    def forward(self, y):
        z = self.HyperEncoder(y)
        return z

class HyperDecoder(nn.Sequential):
    def __init__(self, c_mid, c_out):
        super().__init__()
        self.HyperEncoder = nn.Sequential(
            deconv(c_mid, c_mid),
            nn.ReLU(inplace=True),
            deconv(c_mid, c_mid),
            nn.ReLU(inplace=True),
            conv(c_mid, c_out, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, z_hat):
        y_hat = self.HyperEncoder(z_hat)
        return y_hat

class Spec_hyper(CompressionModel):
    def __init__(self, c_in, c_m, c_out):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(c_m)
        self.gaussian_conditional = GaussianConditional(None)
        self.h_a = HyperEncoder(c_in, c_m)
        self.h_s = HyperDecoder(c_m, c_out)

    def forward(self, y):
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return y_hat, {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}

    def compress(self, y):
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        y_hat = self.gaussian_conditional.decompress(strings=y_strings, indexes=indexes)
        return y_hat, {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings=strings[0], indexes=indexes)
        return y_hat
