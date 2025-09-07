import math

import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers
import Ref_net, Res_net, Spec_net, Predict, Blocks

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))#换底公式，实际上取得是以2为底的log
            for likelihoods_dic in output["likelihoods"].values()
            for likelihoods in likelihoods_dic.values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = 65535 ** 2 * out["mse_loss"] + self.lmbda * out["bpp_loss"]
        return out

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
        return x + x3

class Fusion(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(c_in, c_in * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
        )
        self.ResBlock1 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.ResBlock2 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.ResBlock3 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.ResBlock4 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.ResBlock5 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.ResBlock6 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.ResBlock7 = Blocks.ResBlock(c_in, start=1, slope=0.01)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv3 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv4 = nn.Conv2d(2*c_in, c_in, 3, 1, 1)
        self.conv5 = nn.Conv2d(c_in, c_out, 3, 1, 1)

    def forward(self, prediction):
        pred1 = self.up1(prediction[0])
        pred1 = self.ResBlock1(pred1)
        pred1 = self.conv1(pred1)
        pred1 = self.ResBlock2(pred1)
        pred2 = self.up2(torch.cat((pred1, prediction[1]), 1))
        pred2 = self.ResBlock3(pred2)
        pred2 = self.conv2(pred2)
        pred2 = self.ResBlock4(pred2)
        pred3 = self.up3(torch.cat((pred2, prediction[2]), 1))
        pred3 = self.ResBlock5(pred3)
        pred3 = self.conv3(pred3)
        pred3 = self.ResBlock6(pred3)
        pred4 = self.conv4(torch.cat((pred3, prediction[3]), 1))
        pred4 = self.ResBlock7(pred4)
        out = self.conv5(pred4)
        return out

# class Spec_Refine(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Spec_Refine, self).__init__()
#
#         self.spec_Refine = nn.Sequential(
#             nn.Conv2d(in_channel, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Conv2d(64, out_channel, 3, stride=1, padding=1),
#         )
#     def forward(self, x):
#         fusion_feature = self.spec_Refine(x)
#         return fusion_feature
#
# #参数提取
# class Entropy_parameters(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Entropy_parameters, self).__init__()
#
#         self.entropy_parameters = nn.Sequential(
#             nn.Conv2d(in_channel, 64 * 5 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64 * 5 // 3, 64 * 4 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64 * 4 // 3, out_channel, 1),
#         )
#     def forward(self, params):
#         gaussian_params = self.entropy_parameters(params)
#         return gaussian_params
#
# #重建图像网络
# class FeatureDecoder(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#
#         self.featureDecoder = nn.Sequential(
#             nn.Conv2d(in_channel, 64, 3, stride=1, padding=1),
#             ResidualBlock(64),
#             ResidualBlock(64),
#             nn.Conv2d(64, out_channel, 3, stride=1, padding=1),
#         )
#     def forward(self, x):
#         recon_band = self.featureDecoder(x)
#         return recon_band

#特征提取
class Pre_extract(nn.Module):
    def __init__(self, c_in):
        super(Pre_extract, self).__init__()
        self.Seq = nn.Sequential(
            nn.Conv2d(c_in, 64, 5, stride=2, padding=2),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64))

    def forward(self, x):
        x = self.Seq(x)
        return x

#重建网络
class Reconstruct(nn.Module):
    def __init__(self, c_out):
        super(Reconstruct, self).__init__()
        self.Seq = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, c_out, 5, stride=2, padding=2, output_padding=1))

    def forward(self, x):
        x = self.Seq(x)
        return x

class Post_Process(nn.Module):
    def __init__(self):
        super().__init__()
        self.Seq = nn.Sequential(
            ResidualBlock(8),
            ResidualBlock(8),
            ResidualBlock(8),
            ResidualBlock(8),
            ResidualBlock(8),
            ResidualBlock(8)
        )
    def forward(self, x):
        x = self.Seq(x)
        return x

#预测网络
class Predition_net(CompressionModel):
    def __init__(self):
        super().__init__()

        self.Ref_encoder = Ref_net.I_Encoder(3, 64)
        self.Ref_decoder = Ref_net.I_Decoder2(64, 64)
        self.Ref_hyper = Ref_net.Ref_hyper(64, 64, 64)

        self.Spec_encoder = Spec_net.SpecEncoder(8, 64)
        self.Spec_decoder = Spec_net.SpecDecoder2(64, 64)
        self.Spec_hyper = Spec_net.Spec_hyper(64, 64, 64)

        self.Res_encoder = Res_net.Res_Encoder(8, 64)
        self.Res_decoder = Res_net.Res_Decoder(64, 8)
        self.Res_hyper = Res_net.Res_hyper(64, 64, 64)

        self.Spec_ext = Spec_net.Est_spec_net()
        self.Prediction = Predict.Predict_Net(128, 64)
        self.Prediction2 = Predict.Recon_net(128, 64)
        self.Prediction3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.Prediction_fusion = nn.Conv2d(192, 64, 1, 1, 0)

        self.Fusion = Fusion(64, 8)
        # self.Pre_extract = Pre_extract(8)
        # self.Recon = Reconstruct(8)
        #self.Post = Post_Process()

    def forward(self, input_img):

        ref_y = self.Ref_encoder(input_img[:, 0:3, :, :])
        ref_y_hat, ref_likelihoods = self.Ref_hyper(ref_y)
        ref_x_hat = self.Ref_decoder(ref_y_hat)

        # 特征提取，压缩
        spec = self.Spec_ext(input_img)
        spec_y = self.Spec_encoder(spec)
        spec_y_hat, spec_likelihoods = self.Spec_hyper(spec_y)
        spec_x_hat = self.Spec_decoder(spec_y_hat)

        # 多尺度预测
        prediction = []
        for i in range(0, 4):
            spa_spec_inf = torch.cat((spec_x_hat[i], ref_x_hat[i]), 1)
            pred1 = self.Prediction(spa_spec_inf)
            pred_ref = self.Prediction2(spa_spec_inf)
            pred2 = self.Prediction3(spa_spec_inf)
            pred = self.Prediction_fusion(torch.concat((pred_ref, pred1, pred2), 1))
            prediction.append(pred)
        pred_final = self.Fusion(prediction)
        res = input_img - pred_final
        res_y = self.Res_encoder(res)
        res_y_hat, res_likelihoods = self.Res_hyper(res_y)
        res_x_hat = self.Res_decoder(res_y_hat)
        x_hat = pred_final + res_x_hat

        return {"x_hat": x_hat,
                "ref_x_hat": ref_x_hat,
                "spec_x_hat": spec_x_hat,
                "res_x_hat": res_x_hat,
                "pred": prediction,
                "pred_final": pred_final,
                # "pred1": pred1,
                # "pred2": pred_ref,
                # "pred3": pred2,
                "likelihoods": {"iframe": ref_likelihoods, "feat": spec_likelihoods, "residual": res_likelihoods},
                }


    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = Predition_net()
        net.load_state_dict(state_dict)
        return net

    def compress(self, input_img):

        ref_y = self.Ref_encoder(input_img[:, 0:3, :, :])
        ref_y_hat, ref_out = self.Ref_hyper.compress(ref_y)
        ref_x_hat = self.Ref_decoder(ref_y_hat)

        # 特征提取，压缩
        spec = self.Spec_ext(input_img)
        spec_y = self.Spec_encoder(spec)
        spec_y_hat, spec_out = self.Spec_hyper.compress(spec_y)
        spec_x_hat = self.Spec_decoder(spec_y_hat)

        # 预测，特征级残差压缩
        prediction = []
        for i in range(0, 4):
            spa_spec_inf = torch.cat((spec_x_hat[i], ref_x_hat[i]), 1)
            pred1 = self.Prediction(spa_spec_inf)
            pred_ref = self.Prediction2(spa_spec_inf)
            pred2 = self.Prediction3(spa_spec_inf)
            pred = self.Prediction_fusion(torch.concat((pred_ref, pred1, pred2), 1))
            prediction.append(pred)
        pred_final = self.Fusion(prediction)
        res = input_img - pred_final
        res_y = self.Res_encoder(res)
        _, res_out = self.Res_hyper.compress(res_y)
        # res_x_hat = self.Res_decoder(res_y_hat)
        # x_hat = pred_final + res_x_hat

        return 1, {"strings": {"ref": ref_out["strings"],
                                   "spec": spec_out["strings"],
                                   "res": res_out["strings"]},
                       "shape": {"ref": ref_out["shape"],
                                 "spec": spec_out["shape"],
                                 "res": res_out["shape"]}}


    def decompress(self, strings, shape):

        ref_y_hat = self.Ref_hyper.decompress(strings["ref"], shape["ref"])
        ref_x_hat = self.Ref_decoder(ref_y_hat)

        # 特征
        spec_y_hat = self.Spec_hyper.decompress(strings["spec"], shape["spec"])
        spec_x_hat = self.Spec_decoder(spec_y_hat)

        # 残差
        prediction = []
        for i in range(0, 4):
            spa_spec_inf = torch.cat((spec_x_hat[i], ref_x_hat[i]), 1)
            pred1 = self.Prediction(spa_spec_inf)
            pred_ref = self.Prediction2(spa_spec_inf)
            pred2 = self.Prediction3(spa_spec_inf)
            pred = self.Prediction_fusion(torch.concat((pred_ref, pred1, pred2), 1))
            prediction.append(pred)
        pred_final = self.Fusion(prediction)
        res_y_hat = self.Res_hyper.decompress(strings["res"], shape["res"])
        res_x_hat = self.Res_decoder(res_y_hat)
        x_hat = pred_final + res_x_hat

        return x_hat

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.Ref_hyper.gaussian_conditional,
            "Ref_hyper.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.Spec_hyper.gaussian_conditional,
            "Spec_hyper.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.Res_hyper.gaussian_conditional,
            "Res_hyper.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )

        update_registered_buffers(
            self.Ref_hyper.entropy_bottleneck,
            "Ref_hyper.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.Spec_hyper.entropy_bottleneck,
            "Spec_hyper.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.Res_hyper.entropy_bottleneck,
            "Res_hyper.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = False
        updated |= self.Ref_hyper.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )

        updated |= self.Spec_hyper.entropy_bottleneck.update(force=force)

        updated |= self.Res_hyper.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )

        updated |= self.Ref_hyper.entropy_bottleneck.update(force=force)

        updated |= self.Spec_hyper.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )

        updated |= self.Res_hyper.entropy_bottleneck.update(force=force)

        # updated |= super().update(force=force)
        return updated

