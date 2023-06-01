# -*- coding: utf-8 -*-
"""Radon transformations.

此 Python 代码是与 Radon 变换及其逆变换（逆 Radon 变换）相关的函数集合。
Radon 变换是一种常用于医学成像的工具，更具体地说，是计算机断层扫描 (CT) 中用于将二维图像转换为一系列一维投影的工具。然后逆运算允许从投影重建原始图像。
以下是对最重要功能的简要说明：
pad_image：此函数填充图像，为其添加更多空间。 pad 尺寸是根据原始图像形状的对角线计算的。该函数然后在原始图像周围添加填充，确保图像的新中心与原始中心对齐。这样做是为了确保可以在不丢失信息的情况下转换整个图像。
unpad_image：此函数用于在应用逆 Radon 变换后删除在前一个函数中添加的填充。
get_r_coords：此函数根据给定的直径和点数计算并返回线性间隔的半径坐标数组。
expand_diameter：此函数将直径扩大 K 倍。它确保扩大后的直径为偶数
get_kspace_radial：此函数计算给定参数的 k 空间（傅里叶空间）径向坐标。
radon_transform：此函数将 Radon 变换应用于给定图像。它使用傅里叶模块中的非均匀快速傅里叶变换 (nufft) 将图像变换为其 k 空间表示，然后应用 nufft 的伴随来检索正弦图，这是 Radon 变换的输出。
iradon_transform：此函数将逆 Radon 变换应用于正弦图以检索原始图像。它获取傅立叶滤波器，将 nufft 应用于正弦图，将其乘以滤波器，然后应用 nufft 的伴随来检索图像。然后图像被取消填充并返回。
fft_radon_transform、fft_radon_to_kspace、fft_radon_to_image、fft_kspace_to_sino、fft_sino_to_kspace、fft_discretize_sinogram：这些函数是 Radon 变换的不同实现和变体，它使用快速傅立叶变换 (FFT) 而非非均匀 FFT 进行逆变换。
该代码还使用 jax 和 jax.numpy 来利用 Jax 库提供的自动微分和矢量化功能。
transforms 包是包含 fourier 和 util 模块的外部包。 fourier 模块包括用于快速傅里叶变换及其伴随的函数， util 包括用于调整大小等图像操作的实用函数。
请注意，要运行此代码，您需要正确安装和导入这些模块。还值得一提的是，这段代码是更大系统的一部分，因为它不包含要直接运行的主要功能或脚本。
"""

import numpy as np
import jax
import jax.numpy as jnp
from transforms import fourier, util
import math

__all__ = ["radon_transform", "iradon_transform"]


def get_r_coords(diameter, num):
    if diameter % 2 == 0:
        radius = diameter / 2 - 0.5
        center = -0.5
        return np.linspace(-radius, radius, num) + center
    else:
        radius = (diameter - 1) / 2
        return np.linspace(-radius, radius, num)


def expand_diameter(diameter, K):
    expanded_diameter = int(diameter * K)
    if expanded_diameter % 2 == 1:
        expanded_diameter += 1
    return expanded_diameter


def get_kspace_radial(diameter, expanded_diameter, n_projections):
    r = get_r_coords(diameter, expanded_diameter)
    a = np.linspace(0, np.pi, n_projections, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing="xy")
    x = np.round((r_grid * np.cos(a_grid)) * expanded_diameter / diameter) % expanded_diameter
    y = np.round((-r_grid * np.sin(a_grid)) * expanded_diameter / diameter) % expanded_diameter
    return x.astype(np.int32), y.astype(np.int32)


def radon_transform(image, N=50):
    K = 1.25
    oversamp = 1.25
    width = 4
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, K)
    r = get_r_coords(diameter, expanded_diameter)
    a = np.linspace(0, np.pi, N, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing="xy")
    x = r_grid * np.cos(a_grid)
    y = -r_grid * np.sin(a_grid)

    kspace = fourier.nufft(image, jnp.stack([y, x], axis=-1), oversamp=oversamp, width=width)
    sinogram = fourier.nufft_adjoint(kspace, r[:, None], oshape=kspace.shape[:-1] + (diameter,), oversamp=oversamp, width=width) * diameter / expanded_diameter / np.sqrt(diameter)
    return sinogram.real * diameter


def fft_radon_transform(image, N=50, expansion=6):
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, N)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    image = util.resize(image, oshape)
    kspace = jnp.fft.fft2(jnp.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))
    slices = kspace[..., y.astype(jnp.int32), x.astype(jnp.int32)]
    sinogram = jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.ifftshift(slices, axes=-1), axis=-1), axes=-1)
    # oshape = sinogram.shape[:-1] + (diameter,)
    # sinogram = util.resize(sinogram, oshape)
    return sinogram


def fft_radon_to_kspace(image, expansion=6):
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)

    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    image = util.resize(image, oshape)
    kspace = jnp.fft.fft2(jnp.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))
    return kspace


def fft_radon_to_image(kspace, size):
    image = jnp.fft.fftshift(jnp.fft.ifft2(kspace, axes=(-2, -1)), axes=(-2, -1))
    diagonal = math.ceil(np.sqrt(2) * size)
    oshape = image.shape[:-2] + (diagonal, diagonal)
    image = util.resize(image, oshape)
    return unpad_image(image.real)


def fft_kspace_to_sino(kspace, n_projections, size, expansion):
    diameter = math.ceil(np.sqrt(2.0) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, n_projections)

    slices = kspace[..., y.astype(jnp.int32), x.astype(jnp.int32)]
    sinogram = jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.ifftshift(slices, axes=-1), axis=-1), axes=-1)
    # oshape = sinogram.shape[:-1] + (diameter,)
    # return util.resize(sinogram, oshape)
    return sinogram


def fft_sino_to_kspace(sino, n_projections, size, expansion):
    diameter = math.ceil(np.sqrt(2.0) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, n_projections)

    # oshape = sino.shape[:-2] + (n_projections, expanded_diameter)
    # sino = util.resize(sino, oshape)
    slices = jnp.fft.fftshift(jnp.fft.fft(jnp.fft.ifftshift(sino, axes=-1), axis=-1), axes=-1)
    oshape = sino.shape[:-2] + (expanded_diameter, expanded_diameter)
    kspace = jnp.zeros(oshape, dtype=jnp.complex64).at[..., y.astype(jnp.int32), x.astype(jnp.int32)].set(slices)
    return kspace


def fft_discretize_sinogram(image, sino, expansion=8):
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)
    r = get_r_coords(diameter, expanded_diameter)
    N = sino.shape[-2]
    a = np.linspace(0, np.pi, N, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing="xy")

    circle_y = -r_grid * np.sin(a_grid)
    circle_x = r_grid * np.cos(a_grid)

    round_y = np.round(circle_y * expanded_diameter / diameter) * diameter / expanded_diameter
    round_x = np.round(circle_x * expanded_diameter / diameter) * diameter / expanded_diameter
    round_z = round_x - 1j * round_y
    round_r = np.abs(round_z)
    round_theta = np.angle(round_z)
    round_r = np.where(round_theta < 0, -round_r, round_r)
    round_theta = np.where(round_theta < 0, round_theta + np.pi, round_theta)

    x = np.round((r_grid * np.cos(a_grid)) * expanded_diameter / diameter) % expanded_diameter
    y = np.round((-r_grid * np.sin(a_grid)) * expanded_diameter / diameter) % expanded_diameter

    slices = fourier.nufft(sino, r[:, None], oversamp=1.25, width=4) * np.sqrt(diameter)
    round_theta = (round_theta - a.min()) / (a.max() - a.min()) * (N - 1)
    round_r = (round_r - r.min()) / (r.max() - r.min()) * (expanded_diameter - 1)

    def interp_slices(slices):
        slices = jax.scipy.ndimage.map_coordinates(slices, np.stack([round_theta, round_r], axis=0), order=1, mode="nearest")
        slices = slices.reshape(round_theta.shape)
        return slices

    slices = jax.vmap(interp_slices)(slices)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)

    # image = util.resize(image, oshape)
    # kspace = jnp.fft.fft2(jnp.fft.ifftshift(image, axes=(-2, -1)))
    # kspace = kspace.at[..., y.astype(jnp.int32), x.astype(jnp.int32)].set(slices)
    kspace = jnp.zeros(oshape, dtype=jnp.complex64).at[..., y.astype(jnp.int32), x.astype(jnp.int32)].set(slices)
    return kspace


def pad_image(image):
    diagonal = np.sqrt(2) * max(image.shape[-2:])
    pad = [int(np.ceil(diagonal - s)) for s in image.shape[-2:]]
    new_center = [(s + p) // 2 for s, p in zip(image.shape[-2:], pad)]
    old_center = [s // 2 for s in image.shape[-2:]]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    pad_width = [(0, 0) for i in image.shape[:-2]] + pad_width
    padded_image = jnp.pad(image, pad_width, mode="constant", constant_values=0)
    return padded_image


def unpad_image(image):
    size = int(np.sqrt(image.shape[-1] ** 2 / 2))
    pad_left = (image.shape[-1] - size) // 2
    return image[..., pad_left : pad_left + size, pad_left : pad_left + size]


def get_fourier_filter(diameter, K, oversamp=1.25, width=4):
    size = expand_diameter(diameter, K)
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    r = get_r_coords(diameter, size) / diameter * size
    fourier_filter = 2 * fourier.nufft(jnp.fft.fftshift(f), r[:, None], oversamp=oversamp, width=width).squeeze() * np.sqrt(size)
    return fourier_filter


def iradon_transform(sinogram, K=1.8):
    oversamp = 1.25
    width = 4
    diameter = sinogram.shape[-1]
    expanded_diameter = expand_diameter(diameter, K)
    N = sinogram.shape[-2]
    r = get_r_coords(diameter, expanded_diameter)
    a = np.linspace(0, np.pi, N, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing="xy")
    x = r_grid * np.cos(a_grid)
    y = -r_grid * np.sin(a_grid)
    fourier_filter = get_fourier_filter(diameter, K, oversamp=oversamp, width=width)
    kspace = fourier.nufft(sinogram, r[:, None], oversamp=oversamp, width=width) * np.sqrt(diameter)
    image = (
        fourier.nufft_adjoint(kspace * fourier_filter, jnp.stack([y, x], axis=-1), oshape=sinogram.shape[:-2] + (diameter, diameter), oversamp=oversamp, width=width) * diameter / expanded_diameter
    )
    return unpad_image(image.real / N * np.pi / 2.0)
