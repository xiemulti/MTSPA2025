import os
import time

import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from Predition_net import *
import optimizer
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

#方便计算损失
class my_counter():
    def __init__(self):
        self.eminent = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, eminent):
        self.eminent = eminent
        self.sum += eminent
        self.count += 1
        self.avg = self.sum / self.count

#获取图像
def get_images(DIRECTORY):
    files = os.listdir(DIRECTORY)
    imgDatas = []  # 构造一个存放图片的列表数据结构
    os.chdir(DIRECTORY)
    for file in files:
        img1 = np.fromfile(file, dtype=np.uint8)
        img2 = img1[0::2] * 256 + img1[1::2]
        try:
            img3 = np.uint32(img2.reshape(128, 128, 8))
        except:
            img3 = np.uint32(img2.reshape(512, 512, 8))
        max_number = np.max(img3)
        min_number = np.min(img3)
        img4 = (img3 - min_number) / (max_number - min_number)  # 归一化
        #img4 = img3/65535.0
        imgDatas.append(img4)
    return imgDatas

class myDataset(Dataset):
    def __init__(self, datasource):
        self.datasource = datasource

    def __getitem__(self, index):
        img = self.datasource[index]
        img = torch.FloatTensor(img).permute(2, 0, 1)  # permute改变内部维度顺序
        return img

    def __len__(self):
        return len(self.datasource)

train_dir = r"trainsets"
test_dir = r"testsets"
save_dir = r"models\lambda"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)
epoch_max = 500
train_imgDatas = get_images(train_dir)
test_imgDatas = get_images(test_dir)
our_lambda = 2e5   #率失真优化参数
train_BATCH_SIZE = 16  # 批处理 一次16张 加速 保证数据的有效性 train 对性能有大影响
test_BATCH_SIZE = 2  # 纯加速

pretrain = 0
epoch0 = 0
#初始化网络参数
if pretrain:
    os.chdir('pretrain_model_path')
    model_path = 'model_path.pth.tar'
    net = Predition_net()
    checkpoint = torch.load(model_path)
    model = net.from_state_dict(checkpoint["state_dict"])
    epoch0 = torch.load(model_path)["epoch"] + 1
    model.update(force=True)
    model = model.cuda()
else:
    model = Predition_net().cuda()
    print(model)

if __name__ == '__main__':

    train_data = myDataset(train_imgDatas)
    test_data = myDataset(test_imgDatas)
    train_loader = DataLoader(dataset=train_data, batch_size=train_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    rd = RateDistortionLoss(our_lambda)  # 率失真lmbda#FIXME:lmbda越低，Bpp越低
    rd = rd.cuda()

    optimizer, aux_optimizer = optimizer.configure_optimizers(model, learning_rate=2e-6, aux_learning_rate=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=20, eps=1e-9)

    loss_best = float("inf")

    if pretrain:
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # loss_best = checkpoint['loss']

    #开始训练
    for epoch in range(epoch0, epoch_max + epoch0):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if optimizer.param_groups[0]['lr'] < 5e-9:
            break
        model.train()
        for step, sample in enumerate(train_loader):
            batch_x = sample.cuda()
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            out_net = model(batch_x)
            out_rd1 = rd(out_net, batch_x)
            out_rd1["loss"].backward()
            #防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            if step % 30 == 0:
                print(
                    f"Train epoch {epoch}: ["
                    f"{step*len(batch_x)}/{len(train_loader.dataset)}"
                    f" ({100. * step / len(train_loader):.0f}%)]"
                    f'\tLoss: {out_rd1["loss"].item():.2f} |' #小数点后两位
                    f'\tMSE loss: {out_rd1["mse_loss"].item():.6f} |'
                    f'\tBpp loss: {out_rd1["bpp_loss"].item():.4f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
        # 一轮训练在这里结束
        #之后再算损失，指导网络之后如何学习
        with torch.no_grad():
            for d in test_loader:
                d = d.to('cuda')
                out_net = model(d)
                out_rd2 = rd(out_net, d)
                #计算各部分损失
                loss = my_counter()
                aux_loss = my_counter()
                bpp_loss = my_counter()
                loss.update(out_rd2["loss"])
                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_rd2["bpp_loss"])
                mse_loss = my_counter()
                mse_loss.update(out_rd2["mse_loss"])
            train_log = open(save_dir + '\\train_log.txt', 'a')
            train_log.write(f"Epoch {epoch + 1}   Loss: {loss.avg}   mse_loss:{mse_loss.avg}   "
                            f"bpp_loss:{bpp_loss.avg}   learning_rate:{optimizer.param_groups[0]['lr']}\n")
            train_log.close()
        lr_scheduler.step(loss.avg)

        if loss_best > loss.avg:
            loss_best = loss.avg
            train_log_best = open(save_dir + '\\train_log_best.txt', 'a')
            train_log_best.write(f"Epoch {epoch + 1}   Loss: {loss.avg}   mse_loss:{mse_loss.avg}   "
                                 f"bpp_loss:{bpp_loss.avg}   learning_rate:{optimizer.param_groups[0]['lr']}\n")
            train_log_best.close()
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),  # 目前的网络参数
                "loss": loss.avg,
                "optimizer": optimizer.state_dict(),  # 优化状态
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),  # 指导网络是否能下降
            }
            filename = save_dir + \
                       f"\Predition_net_{our_lambda}_best.pth.tar"
            torch.save(state, filename)


    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),  # 目前的网络参数
        "loss": loss.avg,
        "optimizer": optimizer.state_dict(),  # 优化状态
        "aux_optimizer": aux_optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),  # 指导网络是否能下降
    }
    filename = save_dir + \
               f"\Predition_net_{our_lambda}.pth.tar"
    torch.save(state, filename)
