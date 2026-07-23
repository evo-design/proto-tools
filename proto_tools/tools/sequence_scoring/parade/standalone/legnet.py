# ruff: noqa
# Vendored from https://github.com/autosome-ru/parade @ f8e3e02 (predictor/model/legnet.py).
# Kept byte-for-byte (this header and a trailing newline are the only additions) so PARADE's
# published checkpoints load against the exact classes they were trained with.
from collections import OrderedDict

import torch
from torch import nn
# import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer.
    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.
    """

    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class LegNetClassifier(nn.Module):
    """
    PARADE NN based on LegNet with minor modifications.
    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    conv_sizes : list, optional
        List containing convolution block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    linear_sizes : list, optional
        List containing linear block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.
    """
    __constants__ = ('resize_factor')

    def __init__(self,
                 seqsize,
                 in_channels=6,
                 out_channels=1,
                 conv_sizes=(256, 256, 128, 128, 64, 64, 32, 32),
                 mapper_size=128,
                 linear_sizes=(64,),
                 ks=5,
                 resize_factor=4,
                 use_max_pooling=False,
                 activation=nn.SiLU,
                 final_activation=nn.Identity,
                 filter_per_group=2,
                 se_reduction=4,
                 bn_momentum=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_sizes = conv_sizes
        self.mapper_size = mapper_size
        self.linear_sizes = linear_sizes
        self.resize_factor = resize_factor
        self.use_max_pooling = use_max_pooling
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.bn_momentum = bn_momentum
        seqextblocks = OrderedDict()

        block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                # CHANGE!!!
                out_channels=self.conv_sizes[0],
                kernel_size=ks,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(self.conv_sizes[0],
                           momentum=self.bn_momentum),
            activation()  # Exponential(conv_sizes[0]) #activation()
        )
        seqextblocks['blc0'] = block

        for ind, (prev_sz, sz) in enumerate(zip(self.conv_sizes[:-1], self.conv_sizes[1:])):
            block = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv1d(
                    in_channels=prev_sz,
                    out_channels=sz * self.resize_factor,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz * self.resize_factor), #activation(),

                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=sz * self.resize_factor,
                    kernel_size=ks,
                    groups=sz * self.resize_factor // filter_per_group,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz * self.resize_factor), #activation(),
                SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                # nn.Dropout(0.1),
                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=prev_sz,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(prev_sz,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz), #activation(),
            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * prev_sz,
                    out_channels=sz,
                    kernel_size=ks,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz), #activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        if self.use_max_pooling:
            self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.maxpooling = nn.Identity()

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = block = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=self.conv_sizes[-1],
                out_channels=self.mapper_size,
                kernel_size=1,
                padding='same',
            ),
            # activation()
        )

        self.avgpooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        if self.linear_sizes is not None:
            first_linear = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.mapper_size, self.linear_sizes[0]),
                nn.BatchNorm1d(self.linear_sizes[0],
                               momentum=self.bn_momentum),
                activation()
            )

            linear_blocks = list()
            for prev_sz, sz in zip(self.linear_sizes[:-1], self.linear_sizes[1:]):
                block = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(prev_sz, sz),
                    nn.BatchNorm1d(sz,
                                   momentum=self.bn_momentum),
                    activation()
                )
                linear_blocks.append(block)

            last_linear = nn.Sequential(
                nn.Linear(self.linear_sizes[-1], self.out_channels),
                final_activation()
            )

            self.linear = nn.Sequential(
                nn.Dropout(0.1),
                first_linear,
                *linear_blocks,
                last_linear,
            )
        else:  # i.e. if self.linear_sizes is None
            self.linear = nn.Sequential(
                nn.Linear(self.mapper_size, self.out_channels),
                final_activation()
            )

    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)

        for i in range(len(self.conv_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
            x = self.maxpooling(x)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.mapper(x)
        x = self.avgpooling(x)
        x = self.linear(x)

        # x = F.softmax(x, dim=1)
        # score = (x * self.bins).sum(dim=1)
        return x