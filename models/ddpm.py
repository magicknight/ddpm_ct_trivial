"""
DDPM model.

This code is the FLAX equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py



此代码实现了去噪扩散概率模型 (DDPM) 的一个版本，该模型是一种生成模型，它试图通过随时间扩散数据分布来模拟数据分布，类似于热量或气体随时间扩散的方式。
这用于图像生成等任务，其中模型从学习的数据分布中生成新示例。这里的模型是使用 Flax 实现的，Flax 是一个神经网络库，专为与 JAX 一起使用而设计，JAX 是一个用于高性能数值计算的库。

 更详细地：
 1. 最开始是做一些导入，主要是从Flax、JAX导入必要的模块，从本地模块导入一些辅助函数和层定义。
 2. 定义了一个模型类 DDPM ，代表DDPM的架构。它是 nn.Module 的子类，是Flax中所有神经网络模型的基类。
 3. 定义了模型的 __call__ 函数，它描述了模型的前向传递，即它如何处理输入以生成输出。
    3.1 该函数采用三个参数： x （输入数据）、 labels （与输入数据关联的标签或目标）和 train （指示模型是否处于训练模式的标志）。
    3.2 首先，该函数从 self.config 中检索各种超参数和配置设置。其中包括激活函数、归一化方法和噪声级别列表（“西格玛”）
    3.3 然后，函数检查模型是否有条件。如果是，它将获得时间步长的嵌入，这将用作残差块输入的一部分。
    3.4 之后，准备输入 x 。如果输入数据已经居中，则直接使用；否则，它被重新调整为以 0 为中心。
    3.5 代码的下一部分是“下采样”块。该块的目的是通过一系列残差块处理输入并对数据的空间维度进行下采样（类似于自动编码器中的编码器或 U-Net 中的收缩路径）。
    3.6 下采样完成后，使用“上采样”块。这采用下采样数据并将其空间维度增加回原始大小（类似于自动编码器中的解码器或 U-Net 中的扩展路径）。
    3.7 最后，如果模型配置指示输出应按 sigma 值缩放，则执行此缩放。
4. 返回模型的最终输出。     

从整体架构上看，该模型有点类似于U-Net，有一条下采样路径，一条上采样路径，两条路径之间有skip connections。主要区别在于该模型使用具有可选注意力的残差块，并且可以以扩散时间步长为条件。
"""

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name="ddpm")
class DDPM(nn.Module):
    """DDPM model architecture."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        # config parsing
        config = self.config
        act = get_act(config)
        normalize = get_normalization(config)
        sigmas = utils.get_sigmas(config)

        nf = config.model.nf
        ch_mult = config.model.ch_mult
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        num_resolutions = len(ch_mult)

        AttnBlock = functools.partial(layers.AttnBlock, normalize=normalize)
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, normalize=normalize, dropout=dropout)

        if config.model.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, nf)
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))
        else:
            temb = None

        if config.data.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.0

        # Downsampling block
        hs = [conv3x3(h, nf)]
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                hs.append(h)
            if i_level != num_resolutions - 1:
                hs.append(Downsample(with_conv=resamp_with_conv)(hs[-1]))

        h = hs[-1]
        h = ResnetBlock()(h, temb, train)
        h = AttnBlock()(h)
        h = ResnetBlock()(h, temb, train)

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1), temb, train)
            if h.shape[1] in attn_resolutions:
                h = AttnBlock()(h)
            if i_level != 0:
                h = Upsample(with_conv=resamp_with_conv)(h)

        assert not hs

        h = act(normalize()(h))
        h = conv3x3(h, x.shape[-1], init_scale=0.0)

        if config.model.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = sigmas[labels].reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h
