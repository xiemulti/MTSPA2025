import numpy as np
import torch
from compressai.layers import AttentionBlock, conv1x1
from torch import nn
from torch.nn import GELU
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # input.dim = 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        l1 = self.conv1(self.relu1(input))
        l2 = self.conv2(self.relu2(l1))
        return l2 + input


class Predict_Net(nn.Module):
    def __init__(self, c_in, c_out):
        super(Predict_Net, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1)  # 注意输入维度  6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(64, c_out, kernel_size=3, stride=1, padding=1)  # 输出通道数应该要改  1
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()
        self.res6 = ResBlock()

    def forward(self, input):
        m1 = self.conv1(input)
        m2 = self.res1(m1)
        m3 = self.pool1(m2)
        m4 = self.res2(m3)
        m5 = self.pool2(m4)
        m6 = self.res3(m5)
        m7 = self.res4(m6)
        m8 = m4 + self.up1(m7)
        m9 = self.res5(m8)
        m10 = m2 + self.up2(m9)
        m11 = self.res6(m10)
        m12 = self.relu(self.conv2(m11))
        m13 = self.conv3(m12)
        return m13


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class MSAB2(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type='SW'):
        """ SwinTransformer Block
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.spa_msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.spec_msa = MS_MSA(dim=input_dim, dim_head=head_dim, heads=input_dim//head_dim)
        self.ln3 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.spq_msa(self.ln1(x))+self.spec_msa(self.ln2(x))
        x = x + self.mlp(self.ln3(x))
        return x



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Predict_Net_T(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.enbedding = nn.Conv2d(c_in, 32, 3, 1, 1, bias=False)
        self.MSAB1 = MSAB(dim=32, num_blocks=1, dim_head=32, heads=1)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.MSAB2 = MSAB(dim=64, num_blocks=1, dim_head=32, heads=2)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.MSAB3 = MSAB(dim=128, num_blocks=1, dim_head=32, heads=4)
        self.up1 = nn.ConvTranspose2d(128, 64, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fusion1 = nn.Conv2d(128, 64, 1, 1, bias=False)
        self.MSAB4 = MSAB(dim=64, num_blocks=1, dim_head=32, heads=2)
        self.up2 = nn.ConvTranspose2d(64, 32, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fusion2 = nn.Conv2d(64, 32, 1, 1, bias=False)
        self.MSAB5 = MSAB(dim=32, num_blocks=1, dim_head=32, heads=1)
        self.mapping = nn.Conv2d(32, c_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.enbedding(x)
        x2 = self.MSAB1(x1)
        x3 = self.down1(x2)
        x4 = self.MSAB2(x3)
        x5 = self.down2(x4)
        x6 = self.MSAB3(x5)
        x7 = self.up1(x6)
        x8 = self.fusion1(torch.concat((x4, x7), 1))
        x9 = self.MSAB4(x8)
        x10 = self.up2(x9)
        x11 = self.fusion2(torch.concat((x2, x10), 1))
        x12 = self.MSAB5(x11)
        x13 = self.mapping(x12)
        output = x13 + x
        return output


class Predict_Net_T2(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.enbedding = nn.Conv2d(c_in, 32, 3, 1, 1, bias=False)
        self.MSAB1 = MSAB2(32, 32, 16, 8)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.MSAB2 = MSAB2(64, 64, 16, 8)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.MSAB3 = MSAB2(128, 128, 16, 8)
        self.up1 = nn.ConvTranspose2d(128, 64, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fusion1 = nn.Conv2d(128, 64, 1, 1, bias=False)
        self.MSAB4 = MSAB2(64, 64, 16, 8)
        self.up2 = nn.ConvTranspose2d(64, 32, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fusion2 = nn.Conv2d(64, 32, 1, 1, bias=False)
        self.MSAB5 = MSAB2(32, 32, 16, 8)
        self.mapping = nn.Conv2d(32, c_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.enbedding(x)
        x2 = self.MSAB1(x1)
        x3 = self.down1(x2)
        x4 = self.MSAB2(x3)
        x5 = self.down2(x4)
        x6 = self.MSAB3(x5)
        x7 = self.up1(x6)
        x8 = self.fusion1(torch.concat((x4, x7), 1))
        x9 = self.MSAB4(x8)
        x10 = self.up2(x9)
        x11 = self.fusion2(torch.concat((x2, x10), 1))
        x12 = self.MSAB5(x11)
        x13 = self.mapping(x12)
        output = x13 + x
        return output

class Recon_net(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv_in = nn.Conv2d(c_in, 64, kernel_size=3, padding=1, bias=False)
        self.spec1 = Predict_Net_T(64, 64)
        self.spec2 = Predict_Net_T(64, 64)
        self.conv_out = nn.Conv2d(64, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        h = self.spec1(x)
        h = self.spec2(h)
        x_8 = self.conv_out(h)
        return x_8


class Recon_net2(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv_in = nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False)
        self.spec1 = Predict_Net_T2(32, 32)
        self.spec2 = Predict_Net_T2(32, 32)
        self.conv_out = nn.Conv2d(32, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        h = self.spec1(x)
        h = self.spec2(h)
        x_8 = self.conv_out(h)
        return x_8