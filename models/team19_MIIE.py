"""
NTIRE 2026 Image Denoising Challenge - Network Architectures
Contains: NAFNet, Xformer

Minimal fixes (no structure/naming changes):
  [FIX] ctx.saved_variables → ctx.saved_tensors
  [FIX] torch.meshgrid added indexing='ij'
"""

import math
import numbers
import collections.abc
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

from einops import rearrange


# ==============================================================================
#  Truncated Normal Initialization
# ==============================================================================

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn('mean is more than 2 std from [a, b] in trunc_normal_.')
    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# ==============================================================================
#  Common Utilities
# ==============================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def _ntuple(n):
    from itertools import repeat
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ==============================================================================
#  Normalization Layers
# ==============================================================================

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors  # [FIX] was saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ==============================================================================
#  TLC (Test-time Local Converter) Utilities
# ==============================================================================

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True,
                 fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return (f'kernel_size={self.kernel_size}, base_size={self.base_size}, '
                f'stride={self.kernel_size}, fast_imp={self.fast_imp}')

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1 = min(h - 1, self.kernel_size[0] // r1)
                k2 = min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:]
                       - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = F.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = F.pad(s, (1, 0, 1, 0))
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:]
            s3, s4 = s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2,
                     (h - _h) // 2, (h - _h + 1) // 2)
            out = F.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)
        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


# ==============================================================================
#  NAFNet
# ==============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# ==============================================================================
#  Xformer Components
# ==============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 3, 1, 1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = ChannelAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = (self.qkv(x)
               .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = (attn.view(b_ // nw, nw, self.num_heads, n, n)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
        x = to_3d(x)
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape
        x_size = (Hd, Wd)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, Hd, Wd)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return to_4d(x, h, w)


# ==============================================================================
#  Xformer (ORIGINAL structure - fully compatible with existing YAML & checkpoints)
# ==============================================================================

class Xformer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        img_size=128,
        dim=48,
        num_blocks=[2, 4, 4],
        spatial_num_blocks=[2, 4, 4, 6],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        window_size=[16, 16, 16, 16],
        drop_path_rate=0.1,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
        **kwargs,
    ):
        super(Xformer, self).__init__()
        self.alpha = 1
        self.beta = 1

        #########################################  BCU  #####################################
        self.Convs = nn.ModuleList()
        self.Convs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, stride=1))
        self.Convs.append(nn.Conv2d(dim * 2**2, dim * 2**2, kernel_size=3, padding=1, stride=1))
        self.Convs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, stride=1))
        self.Convs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1))
        self.DWconvs = nn.ModuleList()
        self.DWconvs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, stride=1, groups=dim * 2))
        self.DWconvs.append(nn.Conv2d(dim * 2**2, dim * 2**2, kernel_size=3, padding=1, stride=1, groups=dim * 2**2))
        self.DWconvs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, stride=1, groups=dim * 2))
        self.DWconvs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(spatial_num_blocks))]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        #####################################  channel-wise branch  #####################################
        self.encoder_level1 = nn.Sequential(*[
            ChannelTransformerBlock(dim=dim, num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            ChannelTransformerBlock(dim=int(dim * 2**1), num_heads=heads[1],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(*[
            ChannelTransformerBlock(dim=int(dim * 2**2), num_heads=heads[2],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 2**2))

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            ChannelTransformerBlock(dim=int(dim * 2**2), num_heads=heads[2],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[2])
        ])
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            ChannelTransformerBlock(dim=int(dim * 2**1), num_heads=heads[1],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[1])
        ])
        self.up2_1 = Upsample(int(dim * 2**1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            ChannelTransformerBlock(dim=int(dim), num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[0])
        ])

        #####################################  spatial-wise branch  #####################################
        self.encoder1 = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=dim, input_resolution=(img_size, img_size), num_heads=heads[0],
                window_size=window_size[0],
                shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i],
            ) for i in range(spatial_num_blocks[0])
        ])
        self.d1_2 = Downsample(dim)
        self.encoder2 = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=int(dim * 2**1), input_resolution=(img_size // 2, img_size // 2),
                num_heads=heads[1], window_size=window_size[1],
                shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i],
            ) for i in range(spatial_num_blocks[1])
        ])
        self.d2_3 = Downsample(int(dim * 2**1))
        self.encoder3 = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=int(dim * 2**2), input_resolution=(img_size // 4, img_size // 4),
                num_heads=heads[2], window_size=window_size[2],
                shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i],
            ) for i in range(spatial_num_blocks[2])
        ])
        self.d3_4 = Downsample(int(dim * 2**2))
        self.s_latent = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=int(dim * 2**3), input_resolution=(img_size // 8, img_size // 8),
                num_heads=heads[3], window_size=window_size[3],
                shift_size=0 if (i % 2 == 0) else window_size[3] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:3]):sum(spatial_num_blocks[:4])][i],
            ) for i in range(spatial_num_blocks[3])
        ])

        self.u4_3 = Upsample(int(dim * 2**3))
        self.reduce3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder3 = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=int(dim * 2**2), input_resolution=(img_size // 4, img_size // 4),
                num_heads=heads[2], window_size=window_size[2],
                shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i],
            ) for i in range(spatial_num_blocks[2])
        ])
        self.u3_2 = Upsample(int(dim * 2**2))
        self.reduce2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder2 = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=int(dim * 2**1), input_resolution=(img_size // 2, img_size // 2),
                num_heads=heads[1], window_size=window_size[1],
                shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i],
            ) for i in range(spatial_num_blocks[1])
        ])
        self.u2_1 = Upsample(int(dim * 2**1))
        self.reduce1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder1 = nn.Sequential(*[
            SpatialTransformerBlock(
                dim=int(dim), input_resolution=(img_size, img_size),
                num_heads=heads[0], window_size=window_size[0],
                shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                mlp_ratio=ffn_expansion_factor,
                drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i],
            ) for i in range(spatial_num_blocks[0])
        ])

        #####################################  refinement  #####################################
        self.refinement = nn.Sequential(*[
            ChannelTransformerBlock(
                dim=int(dim * 2**1), num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp)
        out_enc1 = self.encoder1(inp)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc2 = self.d1_2(out_enc1)
        shortcut = inp_enc_level2
        inp_enc_level2 = inp_enc_level2 + self.alpha * self.DWconvs[0](inp_enc2)
        inp_enc2 = inp_enc2 + self.beta * self.Convs[0](shortcut)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc2 = self.encoder2(inp_enc2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc3 = self.d2_3(out_enc2)
        shortcut = inp_enc_level3
        inp_enc_level3 = inp_enc_level3 + self.alpha * self.DWconvs[1](inp_enc3)
        inp_enc3 = inp_enc3 + self.beta * self.Convs[1](shortcut)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc3 = self.encoder3(inp_enc3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc4 = self.d3_4(out_enc3)
        c_latent = self.s_latent(inp_enc_level4)
        s_latent = self.s_latent(inp_enc4)

        inp_dec_level3 = self.up4_3(c_latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec3 = self.u4_3(s_latent)
        inp_dec3 = torch.cat([inp_dec3, out_enc3], 1)
        inp_dec3 = self.reduce3(inp_dec3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec3 = self.decoder3(inp_dec3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec2 = self.u3_2(out_dec3)
        inp_dec2 = torch.cat([inp_dec2, out_enc2], 1)
        inp_dec2 = self.reduce2(inp_dec2)
        shortcut = inp_dec_level2
        inp_dec_level2 = inp_dec_level2 + self.alpha * self.DWconvs[2](inp_dec2)
        inp_dec2 = inp_dec2 + self.beta * self.Convs[2](shortcut)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec2 = self.decoder2(inp_dec2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        inp_dec1 = self.u2_1(out_dec2)
        inp_dec1 = torch.cat([inp_dec1, out_enc1], 1)
        inp_dec1 = self.reduce1(inp_dec1)
        shortcut = inp_dec_level1
        inp_dec_level1 = inp_dec_level1 + self.alpha * self.DWconvs[3](inp_dec1)
        inp_dec1 = inp_dec1 + self.beta * self.Convs[3](shortcut)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec1 = self.decoder1(inp_dec1)

        x = torch.cat([out_dec_level1, out_dec1], 1)
        res = self.refinement(x)

        if self.dual_pixel_task:
            res = res + self.skip_conv(inp)
            res = self.output(res)
        else:
            res = self.output(res) + inp_img

        return res