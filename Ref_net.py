import math
import os
import optimizer
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

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
class I_Encoder(nn.Module):   #c_in=1, c_out=64
    def __init__(self, c_in, c_out):
        super(I_Encoder, self).__init__()

        # scale 2
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.conv1 = nn.Conv2d(c_in, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, c_out, 3, 1, 1)
        self.pooling1 = nn.Conv2d(64, 64, 4, 2, 1)
        self.pooling2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.pooling3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.prelu1 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.block1(x)
        x = self.pooling1(x)
        x = self.block2(x)
        x = self.pooling2(x)
        x = self.block3(x)
        x = self.pooling3(x)
        x = self.block4(x)
        x = self.conv2(x)
        # x = torch.sigmoid(x)
        return x

#参考图像解码器
class I_Decoder(nn.Module):
    def __init__(self, c_in, c_out):        #c_in=1, c_out=64
        super(I_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 64, 3, 1, 1)
        self.block1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block2 = ResidualBlock(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block3 = ResidualBlock(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(16, c_out, 5, 1, 2)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.up3 = nn.PixelShuffle(2)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()

    def forward(self, x):
        xp = self.prelu1(self.conv1(x))
        x = self.block1(xp)
        x = self.conv2(x)
        x = self.prelu2(self.up1(x))
        x = self.prelu3(self.conv3(x))
        x = self.block2(x)
        x = self.conv4(x)
        x = self.prelu4(self.up2(x))
        x = self.prelu5(self.conv5(x))
        x = self.block3(x)
        x = self.conv6(x)
        x = self.prelu6(self.up3(x))
        x = self.conv7(x)
        return x


class I_Decoder2(nn.Module):
    def __init__(self, c_in, c_out):        #c_in=64, c_out=64
        super(I_Decoder2, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 64, 3, 1, 1)
        self.block1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block2 = ResidualBlock(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block3 = ResidualBlock(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(16, c_out, 5, 1, 2)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.up3 = nn.PixelShuffle(2)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()

    def forward(self, x):
        x0 = x
        x = self.prelu1(self.conv1(x))
        x = self.block1(x)
        x = self.conv2(x)
        x = self.prelu2(self.up1(x))
        x1 = self.conv3(x)
        x = self.prelu3(self.block2(x1))
        x = self.conv4(x)
        x = self.prelu4(self.up2(x))
        x2 = self.conv5(x)
        x = self.prelu5(self.block3(x2))
        x = self.conv6(x)
        x = self.prelu6(self.up3(x))
        x3 = self.conv7(x)
        return x0, x1, x2, x3


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


class Parameters(nn.Sequential):
    def __init__(self, c_m):
        super().__init__()
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(c_m * 12 // 3, c_m * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c_m * 10 // 3, c_m * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c_m * 8 // 3, c_m * 6 // 3, 1),
        )

    def forward(self, z_hat):
        param = self.entropy_parameters(z_hat)
        return param


class Ref_hyper(CompressionModel):
    def __init__(self, c_in, c_m, c_out):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(c_m)
        self.gaussian_conditional = GaussianConditional(None)
        self.h_a = HyperEncoder(c_in, c_m)
        self.h_s = HyperDecoder(c_m, c_out)
        # self.context_prediction = CkbMaskedConv2d(
        #     c_m, 2 * c_m, kernel_size=5, padding=2, stride=1
        # )
        # self.parameters = Parameters(c_m)


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

# class Ref_net(CompressionModel):
#     def __init__(self):
#         super().__init__()
#
#         self.Ref_encoder = I_Encoder(3, 24)
#         self.Ref_decoder = I_Decoder(24, 3)
#         self.Ref_hyper = Ref_hyper(24, 24, 24)
#
#     def forward(self, input_img):
#         ref_image = input_img[:, 0:3, :, :]
#         ref_y = self.Ref_encoder(ref_image)
#         ref_y_hat, ref_likelihoods = self.Ref_hyper(ref_y)
#         ref_x_hat = self.Ref_decoder(ref_y_hat)
#         return {"x_hat": ref_x_hat,
#                 "likelihoods": {"iframe": ref_likelihoods},
#                 }
#     def from_state_dict(cls, state_dict):
#         """Return a new model instance from `state_dict`."""
#         net = Ref_net()
#         net.load_state_dict(state_dict)
#         return net
#     def compress(self, input_img):
#         ref_y = self.Ref_encoder(input_img)
#         ref_y_hat, ref_out = self.Ref_hyper.compress(ref_y)
#         ref_x_hat = self.Ref_decoder(ref_y_hat)
#         return ref_x_hat, {"strings": {"ref": ref_out["strings"]},
#                        "shape": {"ref": ref_out["shape"]}}
#
#     def decompress(self, strings, shape):
#         ref_y_hat = self.Ref_hyper.decompress(strings["ref"], shape["ref"])
#         ref_x_hat = self.Ref_decoder(ref_y_hat)
#         return ref_x_hat
#
#     def load_state_dict(self, state_dict):
#         update_registered_buffers(
#             self.Ref_hyper.gaussian_conditional,
#             "Ref_hyper.gaussian_conditional",
#             ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
#             state_dict,
#         )
#         update_registered_buffers(
#             self.Ref_hyper.entropy_bottleneck,
#             "Ref_hyper.entropy_bottleneck",
#             ["_quantized_cdf", "_offset", "_cdf_length"],
#             state_dict,
#         )
#         super().load_state_dict(state_dict)
#
#
#     def update(self, scale_table=None, force=False):
#         if scale_table is None:
#             scale_table = get_scale_table()
#
#         updated = False
#         updated |= self.Ref_hyper.gaussian_conditional.update_scale_table(
#             scale_table, force=force
#         )
#         updated |= self.Ref_hyper.entropy_bottleneck.update(force=force)
#
#         # updated |= super().update(force=force)
#         return updated
#
#
# class RateDistortionLoss(nn.Module):
#     """Custom rate distortion loss with a Lagrangian parameter."""
#     def __init__(self, lmbda):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda
#
#     def forward(self, output, target):
#         N, _, H, W = target.size()
#         out = {}
#         num_pixels = N * H * W
#         out["bpp_loss"] = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))#换底公式，实际上取得是以2为底的log
#             for likelihoods_dic in output["likelihoods"].values()
#             for likelihoods in likelihoods_dic.values()
#         )
#         out["mse_loss"] = self.mse(output["x_hat"], target)
#         out["loss"] = 65535 ** 2 * out["mse_loss"] + self.lmbda * out["bpp_loss"]
#         return out
#
# class my_counter():
#     def __init__(self):
#         self.eminent = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#     def update(self, eminent):
#         self.eminent = eminent
#         self.sum += eminent
#         self.count += 1
#         self.avg = self.sum / self.count
#
# #获取图像
# def get_images(DIRECTORY):
#     files = os.listdir(DIRECTORY)
#     imgDatas = []  # 构造一个存放图片的列表数据结构
#     os.chdir(DIRECTORY)
#     for file in files:
#         img1 = np.fromfile(file, dtype=np.uint8)
#         img2 = img1[0::2] * 256 + img1[1::2]
#         try:
#             img3 = np.uint32(img2.reshape(128, 128, 3))
#         except:
#             img3 = np.uint32(img2.reshape(512, 512, 3))
#         max_number = np.max(img3)
#         min_number = np.min(img3)
#         img4 = (img3 - min_number) / (max_number - min_number)  # 归一化
#         #img4 = img3/65535.0
#         imgDatas.append(img4)
#     return imgDatas
#
# class myDataset(Dataset):
#     def __init__(self, datasource):
#         self.datasource = datasource
#
#     def __getitem__(self, index):
#         img = self.datasource[index]
#         img = torch.FloatTensor(img).permute(2, 0, 1)  # permute改变内部维度顺序
#         return img
#
#     def __len__(self):
#         return len(self.datasource)
#
# train_dir = r"F:\xpj036\8channel\trainsets_3band"
# test_dir = r"F:\xpj036\8channel\testsets_3band"
# save_dir = r"F:\xpj036\Prediction_code\models_ref\2e5\model1"
# if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
# epoch_max = 500
# train_imgDatas = get_images(train_dir)
# test_imgDatas = get_images(test_dir)
# train_mode = "mse"
# our_lambda = 2e5   #率失真优化参数
# train_BATCH_SIZE = 16  # 批处理 一次16张 加速 保证数据的有效性 train 对性能有大影响
# test_BATCH_SIZE = 2  # 纯加速
#
# pretrain = 0
# epoch0 = 0
# #初始化网络参数
# if pretrain:
#     os.chdir(r'F:\xpj036\Prediction_code\models_ref\2e5\model1')
#     model_path = 'Ref_net_200000.0_best.pth.tar'
#     net = Ref_net()
#     checkpoint = torch.load(model_path)
#     model = net.from_state_dict(checkpoint["state_dict"])
#     epoch0 = torch.load(model_path)["epoch"] + 1
#     model.update(force=True)
#     model = model.cuda()
# else:
#     model = Ref_net().cuda()
#     print(model)
#
# if __name__ == '__main__':
#
#     train_data = myDataset(train_imgDatas)
#     test_data = myDataset(test_imgDatas)
#     train_loader = DataLoader(dataset=train_data, batch_size=train_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
#     test_loader = DataLoader(dataset=test_data, batch_size=test_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
#
#     rd = RateDistortionLoss(our_lambda)  # 率失真lmbda#FIXME:lmbda越低，Bpp越低
#     rd = rd.cuda()
#
#     optimizer, aux_optimizer = optimizer.configure_optimizers(model, learning_rate=2e-4, aux_learning_rate=1e-3)
#     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=20, eps=1e-9)
#
#     loss_best = float("inf")
#
#     if pretrain:
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
#         loss_best = checkpoint['loss']
#
#     #开始训练
#     for epoch in range(epoch0, epoch_max + epoch0):
#         print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
#         if optimizer.param_groups[0]['lr'] < 5e-9:
#             break
#         model.train()
#         for step, sample in enumerate(train_loader):
#             batch_x = sample.cuda()
#             optimizer.zero_grad()
#             aux_optimizer.zero_grad()
#             out_net = model(batch_x)
#             out_rd1 = rd(out_net, batch_x)
#             out_rd1["loss"].backward()
#             #防止梯度爆炸
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#             optimizer.step()
#
#             aux_loss = model.aux_loss()
#             aux_loss.backward()
#             aux_optimizer.step()
#             if step % 30 == 0:
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{step*len(batch_x)}/{len(train_loader.dataset)}"
#                     f" ({100. * step / len(train_loader):.0f}%)]"
#                     f'\tLoss: {out_rd1["loss"].item():.2f} |' #小数点后两位
#                     f'\tMSE loss: {out_rd1["mse_loss"].item():.6f} |'
#                     f'\tBpp loss: {out_rd1["bpp_loss"].item():.4f} |'
#                     f"\tAux loss: {aux_loss.item():.2f}"
#                 )
#         # 一轮训练在这里结束
#         #之后再算损失，指导网络之后如何学习
#         with torch.no_grad():
#             for d in test_loader:
#                 d = d.to('cuda')
#                 out_net = model(d)
#                 out_rd2 = rd(out_net, d)
#                 #计算各部分损失
#                 loss = my_counter()
#                 aux_loss = my_counter()
#                 bpp_loss = my_counter()
#                 loss.update(out_rd2["loss"])
#                 aux_loss.update(model.aux_loss())
#                 bpp_loss.update(out_rd2["bpp_loss"])
#                 mse_loss = my_counter()
#                 mse_loss.update(out_rd2["mse_loss"])
#             train_log = open(save_dir + '\\train_log.txt', 'a')
#             train_log.write(f"Epoch {epoch + 1}   Loss: {loss.avg}   mse_loss:{mse_loss.avg}   "
#                             f"bpp_loss:{bpp_loss.avg}   learning_rate:{optimizer.param_groups[0]['lr']}\n")
#             train_log.close()
#         lr_scheduler.step(loss.avg)
#
#         if loss_best > loss.avg:
#             loss_best = loss.avg
#             train_log_best = open(save_dir + '\\train_log_best.txt', 'a')
#             train_log_best.write(f"Epoch {epoch + 1}   Loss: {loss.avg}   mse_loss:{mse_loss.avg}   "
#                                  f"bpp_loss:{bpp_loss.avg}   learning_rate:{optimizer.param_groups[0]['lr']}\n")
#             train_log_best.close()
#             state = {
#                 "epoch": epoch,
#                 "state_dict": model.state_dict(),  # 目前的网络参数
#                 "loss": loss.avg,
#                 "optimizer": optimizer.state_dict(),  # 优化状态
#                 "aux_optimizer": aux_optimizer.state_dict(),
#                 "lr_scheduler": lr_scheduler.state_dict(),  # 指导网络是否能下降
#             }
#             filename = save_dir + \
#                        f"\Ref_net_{our_lambda}_best.pth.tar"
#             torch.save(state, filename)
#
#
#     state = {
#         "epoch": epoch,
#         "state_dict": model.state_dict(),  # 目前的网络参数
#         "loss": loss.avg,
#         "optimizer": optimizer.state_dict(),  # 优化状态
#         "aux_optimizer": aux_optimizer.state_dict(),
#         "lr_scheduler": lr_scheduler.state_dict(),  # 指导网络是否能下降
#     }
#     filename = save_dir + \
#                f"\Ref_net_{our_lambda}.pth.tar"
#     torch.save(state, filename)
