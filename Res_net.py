import torch
from torch import nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers

class Res_Encoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Res_Encoder, self).__init__()
        self.layer1 = nn.Sequential(conv(c_in, 64), GDN(64))
        self.layer2 = nn.Sequential(conv(64, 64), GDN(64))
        self.layer3 = conv(64, c_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = torch.sigmoid(x)
        # x1 = x1 * 255.0  # 量化级别为8
        # x = RoundNoGradient.apply(x1)   # every feature map 0/1
        return x

class Res_Decoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Res_Decoder, self).__init__()
        self.layer1 = deconv(c_in, 64)
        self.layer2 = nn.Sequential(GDN(64, inverse=True), deconv(64, 64))
        self.layer3 = nn.Sequential(GDN(64, inverse=True), deconv(64, c_out))

    def forward(self, x):
        # x = x/255.0  # 量化级别为8
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

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

class Res_hyper(CompressionModel):
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
        # y_hat = self.gaussian_conditional.decompress(strings=y_strings, indexes=indexes)
        return 1, {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings=strings[0], indexes=indexes)
        return y_hat