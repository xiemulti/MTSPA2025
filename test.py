import math
import os
import pickle
import time
import numpy as np
from pytorch_msssim import ms_ssim
import torch
from torch import nn
from Predition_net import *


def bpp_count(bit_strings, size):
    return sum(len(s[0]) for s in bit_strings) * 8 / size

def mse(image0, image1):

    return np.mean(np.square(image1 - image0))

def mse2psnr(mse):

    return 20. * np.log10(65535.) - 10. * np.log10(mse)


def ms_ssim_db(a, b):
    aa = torch.from_numpy(a).permute(2, 0, 1).contiguous()
    aa = aa.unsqueeze(dim=0)
    bb = torch.from_numpy(b).permute(2, 0, 1).contiguous()
    bb = bb.unsqueeze(dim=0)
    msssim = ms_ssim(aa, bb, data_range=65535).numpy().tolist()
    return -10*math.log10(1-msssim)

def evaluation(model, x):#根据compress 输出真实计算的码率
    start = time.time()
    out_enc = model.compress(x)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc[1]["strings"], out_enc[1]["shape"])
    dec_time = time.time() - start
    num_pixels = x.size(2) * x.size(3)
    channel = x.size(1)

    bpp_list = []
    bit_strings = out_enc[1]["strings"]# 改一改
    bpp0 = sum(len(s) for s in bit_strings["ref"][0]) * 8.0 / num_pixels / channel  # 每通道的bpp，方便与JPEG2000对比
    bpp1 = sum(len(s) for s in bit_strings["ref"][1]) * 8.0 / num_pixels / channel
    bpp2 = sum(len(s) for s in bit_strings["spec"][0]) * 8.0 / num_pixels / channel
    bpp3 = sum(len(s) for s in bit_strings["spec"][1]) * 8.0 / num_pixels / channel
    bpp4 = sum(len(s) for s in bit_strings["res"][0]) * 8.0 / num_pixels / channel  # 每通道的bpp，方便与JPEG2000对比
    bpp5 = sum(len(s) for s in bit_strings["res"][1]) * 8.0 / num_pixels / channel  # 每通道的bpp，方便与JPEG2000对比

    bpp_list.append(bpp0)
    bpp_list.append(bpp1)
    bpp_list.append(bpp2)
    bpp_list.append(bpp3)
    bpp_list.append(bpp4)
    bpp_list.append(bpp5)

    return bpp_list, enc_time, dec_time, out_dec[0]

def entropy_estimation_evaluation(model, im):# 根据熵估计码率
    out_net = model.forward(im)

    num_pixels = im.size(1) * im.size(2) * im.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods_dic in out_net["likelihoods"].values()
        for likelihoods in likelihoods_dic.values()
    )

    return bpp.item()


model_path = r'models\Predition_net_200000.0_best.pth.tar'
test_data_path = r"testsets"
net = Predition_net()
model = net.from_state_dict(torch.load(model_path)["state_dict"])
model.update(force=True)
model = model.cuda()
model = model.eval()

Filename_list = []
Esti_bpp_list = []
BPP_list = []
MSE_list = []
MSE2_list = []
PSNR_list = []
PSNR_list2 = []
enc_time_list = []
dec_time_list = []
MS_SSIM_DB_list = []
enctime = []
dectime = []

for filename in os.listdir(test_data_path):#测试集文件位置
    print(filename)
    os.chdir(test_data_path)#进入测试集文件位置
    img1 = np.fromfile(filename, dtype=np.uint8)
    img2 = img1[0::2] * 256 + img1[1::2]
    img3 = np.float32(img2.reshape(512, 512, 8))
    #img4 = img3/65535.0
    max_number = np.max(img3)
    min_number = np.min(img3)
    img4 = (img3 - min_number) / (max_number - min_number)  # 归一化

    im = np.zeros([512, 512, 8], dtype='float32')
    im[:512, :512, :8] = img4
    H, W, C = im.shape
    im = torch.FloatTensor(im).cuda()
    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H, W)

    with (torch.no_grad()):

        Esti_bpp = entropy_estimation_evaluation(model, im)  # 一般来说是准确估计，与后面真实算bpp相互印证，二者应该是相似的
        Esti_bpp_list.append(Esti_bpp)
        bpp, enc_time, dec_time, im_hat = evaluation(model, im)
        im_hat2 = model(im)
        print(filename, f'enc_time:{enc_time}  dec_time:{dec_time}')
        #
        enc_time_list.append(enc_time)
        dec_time_list.append(dec_time)
        #
        output1 = im_hat.cpu().numpy()
        output1 = output1.transpose(1, 2, 0)
        output1[output1 >= 1] = 1
        output1[output1 <= 0] = 0
        recons = output1[:512, :512, :8] * (max_number - min_number) + min_number
        recons1 = np.uint16(recons)
        recons1.tofile(r'recon\recon_image\\' + filename[:-3] + 'raw')

        output2 = im_hat2["x_hat"][0].cpu().numpy()
        output2 = output2.transpose(1, 2, 0)
        output2[output2 >= 1] = 1
        output2[output2 <= 0] = 0
        output2 = output2[:512, :512, :8] * (max_number - min_number) + min_number
        output2 = np.float32(output2)

        #计算各部分压缩性能，包括bpp、psnr、mse、ms-ssim等等
        BPP_list.append(np.sum(bpp))
        recons2 = np.float32(recons)
        our_mse = mse(img3, recons2)
        our_mse2 = mse(img3, output2)
        MSE_list.append(our_mse2/65535/65535)
        PSNR_list.append(mse2psnr(our_mse))
        PSNR_list2.append(mse2psnr(our_mse2))

        #计算ms-ssim
        MS_SSIM_DB_list.append(ms_ssim_db(img3, output2))


print(BPP_list)
print(MSE_list)
print(PSNR_list)
print(PSNR_list2)
# print(MS_SSIM_DB_list)
print('Esti_BPP:', np.mean(Esti_bpp_list))
print('BPP:', np.mean(BPP_list))
print('PSNR:', np.mean(PSNR_list))           #可能存在某些测试集图像无法计算PSNR的现象，多运行几遍可以解决
print('PSNR2:', np.mean(PSNR_list2))
print('压缩平均时间:', np.mean(enc_time_list))
print('解压平均时间:', np.mean(dec_time_list))
# print('MS-SSIM-DB：', np.mean(MS_SSIM_DB_list))
# print(np.mean(enctime))
# print(np.mean(dectime))

