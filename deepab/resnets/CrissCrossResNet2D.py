import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim // 8,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,
                                  out_channels=in_dim // 8,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device
        b, _, h, w = x.shape

        q = self.query_conv(x)
        q_h = rearrange(q, "b c h w -> (b w) h c")
        q_w = rearrange(q, "b c h w -> (b h) w c")

        k = self.key_conv(x)
        k_h = rearrange(k, "b c h w -> (b w) c h")
        k_w = rearrange(k, "b c h w -> (b h) c w")

        v = self.value_conv(x)
        v_h = rearrange(v, "b c h w -> (b w) c h")
        v_w = rearrange(v, "b c h w -> (b h) c w")

        inf = repeat(torch.diag(
            torch.tensor(float("-inf"), device=device).repeat(h), 0),
                     "h1 h2 -> (b w) h1 h2",
                     b=b,
                     w=w)
        e_h = rearrange(torch.bmm(q_h, k_h) + inf,
                        "(b w) h1 h2 -> b h1 w h2",
                        b=b)
        e_w = rearrange(torch.bmm(q_w, k_w), "(b h) w1 w2 -> b h w1 w2", b=b)

        attn = self.softmax(torch.cat([e_h, e_w], 3))
        attn_h, attn_w = attn.chunk(2, dim=-1)
        attn_h = rearrange(attn_h, "b h1 w h2 -> (b w) h1 h2")
        attn_w = rearrange(attn_w, "b h w1 w2 -> (b h) w1 w2")

        out_h = torch.bmm(v_h, rearrange(attn_h, "bw h1 h2 -> bw h2 h1"))
        out_h = rearrange(out_h, "(b w) c h -> b c h w", b=b)
        out_w = torch.bmm(v_w, rearrange(attn_w, "bh w1 w2 -> bh w2 w1"))
        out_w = rearrange(out_w, "(b h) c w -> b c h w", b=b)

        return_attn = torch.stack([
            rearrange(attn_h, "(b w) h1 h2 -> b h2 h1 w", b=b),
            rearrange(attn_w, "(b h) w1 w2 -> b w2 h w1", b=b)
        ],
                                  dim=1)

        return self.gamma * (out_h + out_w) + x, return_attn


class RCCAModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3, return_attn=False):
        super(RCCAModule, self).__init__()
        self.return_attn = return_attn
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      inter_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1),
                      padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
                      bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.cca = CrissCrossAttention(inter_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels,
                      in_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1),
                      padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
                      bias=False), nn.BatchNorm2d(in_channels), nn.ReLU())

    def forward(self, x):
        output = self.conv1(x)
        attns = []
        for _ in range(2):
            output, attn = checkpoint(self.cca, output)
            attns.append(attn)
        output = self.conv2(output)

        if self.return_attn:
            return output, attns
        else:
            return output
