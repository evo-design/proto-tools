"""
Copied from https://github.com/ShenLab-Genomics/SpliceTransformer.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from sinkhorn_transformer import SinkhornTransformer


class ResBlock(nn.Module):

    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1
        # padding calculation:
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class SpEncoder(nn.Module):
    def __init__(self, L, tissue_cnt=11, context_len=4000):
        super(SpEncoder, self).__init__()
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                             21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                              10, 10, 10, 10, 20, 20, 20, 20])
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 3, 1)
        self.tissue_output = nn.Conv1d(L, tissue_cnt, 1)
        self.context_len = context_len

    def forward(self, x, use_usage_head=True):
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out_splice = self.splice_output(skip)
        out_usage = self.tissue_output(skip)
        return torch.concat([out_splice, out_usage], dim=1)

    def forward_feature(self, x):
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        return skip

    def random_mask(self, seq, label):
        if (torch.rand(1)) < 0.05:
            seq[:, :, :self.context_len] *= 0
            seq[:, :, -self.context_len:] *= 0
        return seq, label


class SpEncoder_4tis(nn.Module):
    def __init__(self, L, tissue_cnt=11, context_len=4000):
        super(SpEncoder_4tis, self).__init__()
        #
        # convolution window size in residual units
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                             21, 21, 21, 21, 21, 21, 21, 21])
        # atrous rate in residual units
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                              10, 10, 10, 10, 20, 20, 20, 20])
        #
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)
        self.context_len = context_len

    def forward(self, x, use_usage_head=True):
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out = [
            F.softmax(self.conv_last1(skip), dim=1),
            torch.sigmoid(self.conv_last2(skip)),
            F.softmax(self.conv_last3(skip), dim=1),
            torch.sigmoid(self.conv_last4(skip)),
            F.softmax(self.conv_last5(skip), dim=1),
            torch.sigmoid(self.conv_last6(skip)),
            F.softmax(self.conv_last7(skip), dim=1),
            torch.sigmoid(self.conv_last8(skip))]
        return torch.cat(out, 1)

    def forward_feature(self, x):
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        return skip


class AttnBlock(nn.Module):
    def __init__(self, dim=256, depth=6, causal=False, max_seq_len=8192, reversible=False) -> None:
        super().__init__()
        bucket_size = 64
        axial_position_shape = ((max_seq_len // bucket_size), bucket_size)
        self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
        # self.emb_dropout = nn.Dropout(0.05)
        self.attn = SinkhornTransformer(
            dim, depth,
            heads=8,
            n_local_attn_heads=2,
            max_seq_len=max_seq_len,
            attn_layer_dropout=0.1,
            layer_dropout=0.1,
            ff_dropout=0.1,
            ff_chunks=10, causal=causal, reversible=reversible, non_permutative=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = torch.transpose(x, 1, 2).contiguous()
        x = x + self.pos_emb(x)
        # x = self.emb_dropout(x)
        x = self.attn(x)
        x = self.norm(x)
        x = torch.transpose(x, 1, 2).contiguous()
        return x


class SpEncoder_L(nn.Module):
    def __init__(self, L, tissue_cnt=15, context_len=4000):
        super(SpEncoder_L, self).__init__()
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                             21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                              10, 10, 10, 10, 20, 20, 20, 20])
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 3, 1)
        self.tissue_output = nn.Conv1d(L, tissue_cnt, 1)
        self.context_len = context_len

    def forward(self, x, feature=False):
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        if feature:
            return skip
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out_splice = self.splice_output(skip)
        out_usage = self.tissue_output(skip)
        return torch.concat([out_splice, out_usage], dim=1)


class SpEncoder2_L(nn.Module):
    def __init__(self, L, tissue_cnt=4, context_len=4000):
        super(SpEncoder2_L, self).__init__()
        #
        # convolution window size in residual units
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                             21, 21, 21, 21, 21, 21, 21, 21])
        # atrous rate in residual units
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                              10, 10, 10, 10, 20, 20, 20, 20])
        #
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.context_len = context_len

    def forward(self, x, feature=False):
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        if feature:
            return skip
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out = [
            F.softmax(self.conv_last0(skip), dim=1),
            torch.sigmoid(self.conv_last1(skip)),
        ]
        return torch.cat(out, 1)


class SpTransformer(nn.Module):
    def __init__(self, dim=32, context_len=4000, tissue_num=15, max_seq_len=8192, attn_depth=6, training=False) -> None:
        super().__init__()
        self.context_len = context_len
        self.encoderL = 128 + 64
        self.tissue_num = tissue_num
        self.encoder = self.load_pretrain(training)
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, dim, 1),
            nn.Conv1d(dim, dim, 1),
        )
        self.conv2 = nn.Conv1d(dim+self.encoderL, dim*2, 1)
        #
        self.max_seq_len = max_seq_len
        self.attn = AttnBlock(
            dim*2, depth=attn_depth, max_seq_len=self.max_seq_len)
        self.splice = nn.Conv1d(dim*2, 3, 1)
        self.usage = nn.Conv1d(dim*2, tissue_num, 1)

    def load_pretrain(self, training=False):
        model1 = SpEncoder_L(128, tissue_cnt=self.tissue_num,
                             context_len=self.context_len)
        model2 = SpEncoder2_L(64, tissue_cnt=self.tissue_num,
                              context_len=self.context_len)
        if training:
            print('Load previous model SpEncoder1')
            WEIGHTS = '/public/home/shenninggroup/yny/code/CellSplice/runs/T-EncoderL1-1000-128_3/best.ckpt'
            save_dict = torch.load(
                WEIGHTS, map_location=torch.device('cpu'))
            model1.load_state_dict(save_dict["state_dict"], strict=True)
            print('Load previous model SpEncoder2')
            WEIGHTS = '/public/home/shenninggroup/yny/code/Splice-Pytorch/model/stage1.ckpt'
            model2.load_state_dict(torch.load(
                WEIGHTS, map_location=torch.device('cpu')), strict=False)
            print('Done')
        return nn.ModuleList([model1, model2])

    def forward(self, x):
        target_output_len = x.size(2) - 2 * self.context_len
        target_mid_len = self.max_seq_len
        odd_fix = x.size(2) & 1
        #
        with torch.no_grad():
            feat1 = [self.encoder[i](x, feature=True) for i in [0, 1]]
        feat1 = torch.concat(feat1, dim=1)
        #
        feat2 = self.conv1(x)
        # clip 1
        seq_len = x.size(2)
        feat1 = F.pad(feat1, (-(seq_len-target_mid_len) //
                      2, -(seq_len-target_mid_len)//2+odd_fix))
        feat2 = F.pad(feat2, (-(seq_len-target_mid_len) //
                      2, -(seq_len-target_mid_len)//2+odd_fix))
        #
        emb = torch.concat([feat1, feat2], dim=1)
        emb = self.conv2(emb)
        attn = self.attn(emb)

        splice_out = self.splice(attn)
        usage_out = self.usage(attn)

        out = torch.concat([splice_out, usage_out], dim=1)

        seq_len = out.size(2)
        out = F.pad(out, (-(seq_len-target_output_len) //
                    2+odd_fix, -(seq_len-target_output_len)//2))
        return out
