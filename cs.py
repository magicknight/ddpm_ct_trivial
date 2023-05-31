"""
该代码似乎是一个更大的框架的一部分，该框架用于处理和重建来自计算机断层扫描 (CT) 扫描、磁共振成像 (MRI) 扫描和稀疏 MAR 或使用 noise2inverse 的金属伪影减少图像的图像——一种训练神经网络的方法执行图像去噪和重建任务。
此 Python 代码旨在对图像执行各种类型的采样。它相当复杂，涉及深度学习模型、随机微分方程 (SDE) 和图像变换技术，所有这些都在 JAX 中实现，JAX 是 Google 的一个数值计算库。

get_cartesian_mask(shape, n_keep=30)：此函数为具有 n_keep 定义的特定欠采样率的MRI图像创建掩码。欠采样遵循笛卡尔采样的模式，其中 k 空间的中心（图像的傅里叶变换）比外围更密集地采样。
segment_metal(image, smear_size=3)：此函数用于检测图像中的金属伪影，使用固定的亨斯菲尔德单位 (HU) 阈值 3300。然后使用最大池化操作涂抹金属伪影。
get_metal_trace(image, projection=50, expansion=6)：此函数将氡变换（投影或线积分）应用于图像以识别金属的存在。
get_ct_mask(size, n_angles, expansion)：此函数通过计算径向图案中的 k 空间坐标来为 CT 图像创建掩模。
get_masks(config, img)：此函数根据 config 对象选择适当的掩码（CT、MRI 或 MAR）。
get_known(config, img)：此函数根据任务类型（CT、MRI 或 MAR）从图像中提取已知数据。
merge_known_with_mask(config, x_space, known, mask, coeff=1.)：此函数将已知数据与欠采样掩码合并。合并的程度由参数 coeff 控制。
get_projection_sampler(config, sde, model, shape, predictor, corrector, inverse_scaler, n_steps=1, probability_flow=False, continuous=True, denoise=True, eps=1e-5)：此函数创建一个采样函数，用于从学习模型。
它根据图像是 CT 扫描、MRI 还是 MAR 采用不同的变换。
projection_sampler(rng, state, img, coeff, snr)：这个函数定义在 get_projection_sampler 里面，从学习模型中生成样本。该函数使用随机微分方程 (SDE) 的预测-校正方法来生成样本。
get_inpaint_update_fn(update_fn)：该函数也在 get_projection_sampler 内部定义，修改模型的更新函数，使其在更新步骤中考虑已知数据。
get_baseline_sampler ：该函数根据提供的配置、SDE、模型和其他参数创建采样器函数。此处生成的采样器将根据配置中指定的特定采样任务（'mri'、'ct'、'mar'、'sparse_mar'）执行基线采样过程。
对于每种类型的任务，定义了适当的函数 to_space 和 from_space 以将图像转换到采样任务的适当域，然后返回。如果提供了无法识别的任务，则会抛出 ValueError。
get_langevin_sampler 和 get_langevin_projection_sampler ：这些函数的工作原理与 get_baseline_sampler 类似，但它们分别执行朗之万采样和朗之万投影采样。 
Langevin 采样是深度学习中常用的一种方法，用于从复杂的高维分布中采样，尤其是在生成模型的背景下。
get_cs_solver ：该函数生成一个执行压缩感知（CS）采样的函数。根据配置，它可以执行不同类型的采样，如 Projection、Langevin 和 Langevin Projection。压缩感知是一种信号处理技术，可以从少量线性、非自适应测量中重建信号。


这段代码的核心思想是将 CT、MRI 或 MAR 图像转换成机器学习模型可以处理的形式（使用傅里叶和拉东变换），然后使用训练好的模型重建图像。 noise2inverse 方法用于处理图像中存在的噪声。
这些功能中的每一个都使用许多共享的子例程和实用程序来执行它们的任务。例如， loop_body 函数用作采样过程的主要迭代， score_fn 用于计算给定样本的对数概率的“分数”或梯度，这是采样过程的关键部分.
此代码需要对机器学习和图像处理技术有很好的理解，并熟悉 JAX 库以进行高效的数值计算。它可能是用于医学图像分析或类似任务的更大系统的一部分。

"""


from models import utils as mutils
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as random
from sampling import NoneCorrector, NonePredictor, ReverseDiffusionPredictor, get_predictor, get_corrector, shared_predictor_update_fn, shared_corrector_update_fn
from utils import batch_mul
import functools
import math
from transforms.radon import (
    radon_transform,
    get_r_coords,
    expand_diameter,
    fft_radon_to_kspace,
    fft_radon_to_image,
    fft_discretize_sinogram,
    fft_radon_transform,
    fft_kspace_to_sino,
    fft_sino_to_kspace,
    get_kspace_radial,
)
from mar.create_artifacts import convert_HU_to_png, convert_png_to_HU
from transforms.radon import radon_transform, iradon_transform
import flax.linen as nn


def get_cartesian_mask(shape, n_keep=30):
    # shape [Tuple]: (H, W)
    size = shape[0]
    center_fraction = n_keep / 1000
    acceleration = size / n_keep

    num_rows, num_cols = shape[0], shape[1]
    num_low_freqs = int(round(num_cols * center_fraction))

    # create the mask
    mask = np.zeros((num_rows, num_cols), dtype=np.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[:, pad : pad + num_low_freqs] = True

    # determine acceleration rate by adjusting for the number of low frequencies
    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)

    offset = round(adjusted_accel) // 2

    accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
    accel_samples = np.around(accel_samples).astype(np.uint32)
    mask[:, accel_samples] = True

    return mask


def get_cartesian_mask_coordinates(size, n_keep):
    i, j = np.nonzero(get_cartesian_mask((size, size), n_keep))
    y_grid = i.reshape((n_keep, -1))
    x_grid = j.reshape((n_keep, -1))
    return x_grid, y_grid


def segment_metal(image, smear_size=3):
    assert image.shape[-1] == 1
    metal_mark = convert_png_to_HU(image) > 3300.0
    metal_mark = nn.max_pool(metal_mark.astype(jnp.float32), (smear_size, smear_size), strides=(1, 1), padding="SAME")
    return metal_mark


def get_metal_trace(image, projection=50, expansion=6):
    # shape of image: B x H x W
    metal = segment_metal(image[..., None]).astype(jnp.float32)[..., 0]
    metal_trace = fft_radon_transform(metal, N=projection, expansion=expansion).real
    metal_trace = metal_trace > 5.0
    return metal_trace


def get_ct_mask(size, n_angles, expansion):
    diameter = math.ceil(np.sqrt(2.0) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, n_angles)
    return jnp.zeros((expanded_diameter, expanded_diameter)).at[y, x].set(1.0)


def get_ct_subsampling_mask(size, n_angles, expansion):
    diameter = math.ceil(np.sqrt(2.0) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    sampled_row_ids = np.round(np.linspace(0, size - 1, n_angles)).astype(np.int32)
    return jnp.zeros((size, expanded_diameter)).at[sampled_row_ids, :].set(1.0)


def get_masks(config, img):
    if config.sampling.task == "ct":
        mask = get_ct_subsampling_mask(config.data.image_size, n_angles=config.sampling.n_projections, expansion=config.sampling.expansion)[None, ..., None]
        return mask

    elif config.sampling.task == "mri":
        mask = get_cartesian_mask((config.data.image_size, config.data.image_size), n_keep=config.sampling.n_projections)
        mask = mask[None, :, :, None].astype(jnp.float32)
        return mask

    elif config.sampling.task in ("sparse_mar", "mar"):
        if config.sampling.task == "mar":
            n_projections = config.data.image_size
        else:
            n_projections = config.sampling.n_projections

        mask1 = (~get_metal_trace(img[..., 0], projection=config.data.image_size, expansion=config.sampling.expansion)[..., None]).astype(jnp.float32)
        mask2 = get_ct_subsampling_mask(config.data.image_size, n_angles=n_projections, expansion=config.sampling.expansion)[None, ..., None]
        return mask1 * mask2

    else:
        raise ValueError(f"task {config.sampling.mask} not recognized.")


def get_known(config, img):
    if config.sampling.task in ("ct", "mar", "sparse_mar"):
        n_projections = config.data.image_size
        if config.sampling.task in ("mar", "sparse_mar"):
            metal = segment_metal(img)
            img = jnp.where(metal, 0.0, img)
        sinogram = radon_transform(img[..., 0], n_projections)
        known = fft_discretize_sinogram(img[..., 0], sinogram, config.sampling.expansion)
        return known[..., None]

    elif config.sampling.task == "mri":
        return get_kspace(img, axes=(1, 2))

    else:
        raise ValueError(f"task {config.sampling.mask} not recognized.")


def merge_known_with_mask(config, x_space, known, mask, coeff=1.0):
    if config.sampling.task == "mri":
        return known * mask * coeff + x_space * (1.0 - mask * coeff)
    if config.sampling.task in ("ct", "mar", "sparse_mar"):
        size = config.data.image_size
        expansion = config.sampling.expansion
        x_sino = fft_kspace_to_sino(x_space[..., 0], size, size, expansion)[..., None]
        known_sino = fft_kspace_to_sino(known[..., 0], size, size, expansion)[..., None]
        merged_sino = x_sino * (1.0 - mask * coeff) + known_sino * mask * coeff
        merged_kspace = fft_sino_to_kspace(merged_sino[..., 0], size, size, expansion)[..., None]
        ct_mask = get_ct_mask(size, size, expansion)[None, ..., None]
        merged_kspace = merged_kspace * ct_mask + x_space * (1.0 - ct_mask)
        return merged_kspace

    else:
        raise ValueError(f"task {config.sampling.mask} not recognized.")


def get_kspace(img, axes):
    shape = img.shape[axes[0]]
    return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(img, axes=axes), axes=axes), axes=axes) / shape


def kspace_to_image(kspace, axes):
    shape = kspace.shape[axes[0]]
    return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes) * shape


def get_projection_sampler(config, sde, model, shape, predictor, corrector, inverse_scaler, n_steps=1, probability_flow=False, continuous=True, denoise=True, eps=1e-5):
    if config.sampling.task == "mri":
        to_space = lambda x: get_kspace(x, (1, 2))
        from_space = lambda x: kspace_to_image(x, (1, 2)).real

    elif config.sampling.task in ("ct", "mar", "sparse_mar"):
        to_space = lambda x: fft_radon_to_kspace(x[..., 0], config.sampling.expansion)[..., None]
        from_space = lambda x: fft_radon_to_image(x[..., 0], config.data.image_size)[..., None]

    else:
        raise ValueError(f"Task {config.sampling.task} not recognized.")

    def get_inpaint_update_fn(update_fn):
        def inpaint_update_fn(rng, state, x, t, mask, known, coeff):
            x_space = to_space(x)

            mean, std = sde.marginal_prob(known, t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            noise_space = to_space(noise)
            noisy_known = mean + batch_mul(std, noise_space)

            x_space = merge_known_with_mask(config, x_space, noisy_known, mask, coeff)
            x = from_space(x_space)

            rng, step_rng = jax.random.split(rng)
            x, x_mean = update_fn(step_rng, state, x, t)

            return x

        return inpaint_update_fn

    def projection_sampler(rng, state, img, coeff, snr):
        # Initial sample
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)

        mask = get_masks(config, img)
        known = get_known(config, img)

        predictor_update_fn = functools.partial(shared_predictor_update_fn, sde=sde, model=model, predictor=predictor, probability_flow=probability_flow, continuous=continuous)
        corrector_update_fn = functools.partial(shared_corrector_update_fn, sde=sde, model=model, corrector=corrector, continuous=continuous, snr=snr, n_steps=n_steps)

        cs_predictor_update_fn = get_inpaint_update_fn(predictor_update_fn)
        cs_corrector_update_fn = get_inpaint_update_fn(corrector_update_fn)

        timesteps = jnp.linspace(sde.T, eps, sde.N)

        def loop_body(carry, i):
            rng, x = carry
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t
            rng, step_rng = random.split(rng)
            x = cs_corrector_update_fn(step_rng, state, x, vec_t, mask, known, coeff)
            rng, step_rng = random.split(rng)
            x = cs_predictor_update_fn(step_rng, state, x, vec_t, mask, known, coeff)
            output = x
            return (rng, x), output

        _, all_samples = jax.lax.scan(loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)

        output = all_samples[-1]
        # output = all_samples
        if denoise:
            t_eps = jnp.full((output.shape[0],), eps)
            k, std = sde.marginal_prob(jnp.ones_like(output), t_eps)
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous, return_state=False)
            score = score_fn(output, t_eps)
            output = output / k + batch_mul(std**2, score / k)
            output_space = to_space(output)
            output_space = merge_known_with_mask(config, output_space, known, mask, 1.0)
            output = from_space(output_space)

        return inverse_scaler(output)

    return jax.pmap(projection_sampler, axis_name="batch", in_axes=(0, 0, 0, None, None))


def get_baseline_sampler(config, sde, model, shape, predictor, corrector, inverse_scaler, n_steps=1, probability_flow=False, continuous=True, denoise=True, eps=1e-5):
    if config.sampling.task == "mri":

        def to_space(x):
            kspace = get_kspace(x, (1, 2))
            return jnp.concatenate([kspace.real, kspace.imag], axis=-1)

        def from_space(x):
            return kspace_to_image(x[..., 0] + x[..., 1] * 1j, (1, 2)).real

    elif config.sampling.task in ("ct", "mar", "sparse_mar"):
        to_space = lambda x: radon_transform(x[..., 0], config.data.image_size)[..., None]
        from_space = lambda x: iradon_transform(x[..., 0], config.sampling.iradon_K)[..., None]

    else:
        raise ValueError(f"Task {config.sampling.task} not recognized.")

    def likelihood(rng, x, t, mask, known, projection_sigma_rate):
        model_space = to_space(x)
        rng, step_rng = random.split(rng)
        noise_space = to_space(random.normal(step_rng, x.shape)).reshape((x.shape[0], -1))

        model_space = model_space.reshape((x.shape[0], -1))
        known = known.reshape((x.shape[0], -1))
        mask = mask.reshape((-1, known.shape[1]))
        mean, std = sde.marginal_prob(jnp.ones_like(known), t)
        effective_std = projection_sigma_rate * std[:, None]

        given_space = known * mean + batch_mul(std, noise_space)
        squared_dist = jnp.sum(mask * jnp.square(model_space - given_space) / (2 * effective_std**2), axis=-1)
        log_prob = -squared_dist - jnp.sum(mask * jnp.log(2 * np.pi * effective_std**2), axis=-1) / 2
        return log_prob.sum()

    likelihood_grad_fn = jax.grad(likelihood, argnums=1)

    def predictor_update_fn(rng, state, x, t, mask, known, projection_sigma_rate):
        score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)

        rng, step_rng = random.split(rng)

        def total_grad_fn(x, t):
            return score_fn(x, t) + likelihood_grad_fn(step_rng, x, t, mask, known, projection_sigma_rate)

        if predictor is None:
            predictor_obj = NonePredictor(sde, total_grad_fn, probability_flow)
        else:
            predictor_obj = predictor(sde, total_grad_fn, probability_flow)
        return predictor_obj.update_fn(rng, x, t)

    def corrector_update_fn(rng, state, x, t, mask, known, projection_sigma_rate, snr):
        score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)

        rng, step_rng = random.split(rng)

        def total_grad_fn(x, t):
            return score_fn(x, t) + likelihood_grad_fn(step_rng, x, t, mask, known, projection_sigma_rate)

        if corrector is None:
            corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
        return corrector_obj.update_fn(rng, x, t)

    def baseline_sampler(rng, state, img, projection_sigma_rate, snr):
        # Initial sample
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)
        mask = get_masks(config, img)
        if config.sampling.task == "mri":
            mask = jnp.tile(mask, (1, 1, 1, 2))
        known = to_space(img)

        timesteps = jnp.linspace(sde.T, eps, sde.N)

        def loop_body(carry, i):
            rng, x = carry
            t = timesteps[i]
            vec_t = jnp.full(shape[0], t)
            rng, step_rng = random.split(rng)
            x, _ = corrector_update_fn(step_rng, state, x, vec_t, mask, known, projection_sigma_rate, snr)
            rng, step_rng = random.split(rng)
            x, _ = predictor_update_fn(step_rng, state, x, vec_t, mask, known, projection_sigma_rate)
            return (rng, x), x

        _, all_samples = jax.lax.scan(loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)

        output = all_samples[-1]
        if denoise:
            t_eps = jnp.full((output.shape[0],), eps)
            k, std = sde.marginal_prob(jnp.ones_like(output), t_eps)
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous, return_state=False)
            score = score_fn(output, t_eps)
            output = output / k + batch_mul(std**2, score / k)

        return inverse_scaler(output)

    return jax.pmap(baseline_sampler, axis_name="batch", in_axes=(0, 0, 0, None, None))


def get_langevin_sampler(config, sde, model, shape, corrector, inverse_scaler, n_steps=1, continuous=True, denoise=True, eps=1e-5):
    if config.sampling.task == "mri":

        def to_space(x):
            kspace = get_kspace(x, (1, 2))
            return jnp.concatenate([kspace.real, kspace.imag], axis=-1)

        def from_space(x):
            return kspace_to_image(x[..., 0] + x[..., 1] * 1j, (1, 2)).real

    elif config.sampling.task in ("ct", "mar", "sparse_mar"):
        to_space = lambda x: radon_transform(x[..., 0], config.data.image_size)[..., None]
        from_space = lambda x: iradon_transform(x[..., 0], config.sampling.iradon_K)[..., None]

    else:
        raise ValueError(f"Task {config.sampling.task} not recognized.")

    def likelihood(x, t, mask, known, projection_sigma_rate):
        model_space = to_space(x)
        model_space = model_space.reshape((x.shape[0], -1))
        known = known.reshape((x.shape[0], -1))
        mask = mask.reshape((-1, known.shape[1]))
        mean, std = sde.marginal_prob(jnp.ones_like(known), t)
        effective_std = projection_sigma_rate * std[:, None]
        given_space = known * mean
        squared_dist = jnp.sum(mask * jnp.square(model_space - given_space) / (2 * effective_std**2), axis=-1)
        log_prob = -squared_dist - jnp.sum(mask * jnp.log(2 * np.pi * effective_std**2), axis=-1) / 2
        return log_prob.sum()

    likelihood_grad_fn = jax.grad(likelihood, argnums=0)

    def langevin_corrector_update_fn(rng, state, x, t, mask, known, projection_sigma_rate, snr):
        score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)

        def total_grad_fn(x, t):
            return score_fn(x, t) + likelihood_grad_fn(x, t, mask, known, projection_sigma_rate)

        if corrector is None:
            corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
        return corrector_obj.update_fn(rng, x, t)

    def langevin_sampler(rng, state, img, projection_sigma_rate, snr):
        # Initial sample
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)
        mask = get_masks(config, img)
        if config.sampling.task == "mri":
            mask = jnp.tile(mask, (1, 1, 1, 2))
        known = to_space(img)

        timesteps = jnp.linspace(sde.T, eps, sde.N)

        def loop_body(carry, i):
            rng, x = carry
            t = timesteps[i]
            vec_t = jnp.full(shape[0], t)
            rng, step_rng = random.split(rng)
            x, _ = langevin_corrector_update_fn(step_rng, state, x, vec_t, mask, known, projection_sigma_rate, snr)
            return (rng, x), x

        _, all_samples = jax.lax.scan(loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)

        output = all_samples[-1]
        if denoise:
            t_eps = jnp.full((output.shape[0],), eps)
            k, std = sde.marginal_prob(jnp.ones_like(output), t_eps)
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous, return_state=False)
            score = score_fn(output, t_eps)
            output = output / k + batch_mul(std**2, score / k)

        return inverse_scaler(output)

    return jax.pmap(langevin_sampler, axis_name="batch", in_axes=(0, 0, 0, None, None))


def get_langevin_projection_sampler(config, sde, model, shape, corrector, inverse_scaler, n_steps=1, continuous=True, denoise=True, eps=1e-5):
    if config.sampling.task == "mri":
        to_space = lambda x: get_kspace(x, (1, 2))
        from_space = lambda x: kspace_to_image(x, (1, 2)).real

    elif config.sampling.task in ("ct", "mar", "sparse_mar"):
        to_space = lambda x: fft_radon_to_kspace(x[..., 0], config.sampling.expansion)[..., None]
        from_space = lambda x: fft_radon_to_image(x[..., 0], config.data.image_size)[..., None]

    else:
        raise ValueError(f"Task {config.sampling.task} not recognized.")

    def get_inpaint_update_fn(update_fn):
        def inpaint_update_fn(rng, state, x, t, mask, known, coeff):
            x_space = to_space(x)

            mean, std = sde.marginal_prob(known, t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            noise_space = to_space(noise)
            noisy_known = mean + batch_mul(std, noise_space)

            x_space = merge_known_with_mask(config, x_space, noisy_known, mask, coeff)
            x = from_space(x_space)

            rng, step_rng = jax.random.split(rng)
            x, x_mean = update_fn(step_rng, state, x, t)

            return x

        return inpaint_update_fn

    def langevin_projection_sampler(rng, state, img, coeff, snr):
        # Initial sample
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)

        mask = get_masks(config, img)
        known = get_known(config, img)

        corrector_update_fn = functools.partial(shared_corrector_update_fn, sde=sde, model=model, corrector=corrector, continuous=continuous, snr=snr, n_steps=n_steps)

        cs_corrector_update_fn = get_inpaint_update_fn(corrector_update_fn)

        timesteps = jnp.linspace(sde.T, eps, sde.N)

        def loop_body(carry, i):
            rng, x = carry
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t
            rng, step_rng = random.split(rng)
            x = cs_corrector_update_fn(step_rng, state, x, vec_t, mask, known, coeff)
            output = x
            return (rng, x), output

        _, all_samples = jax.lax.scan(loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)

        output = all_samples[-1]
        if denoise:
            t_eps = jnp.full((output.shape[0],), eps)
            k, std = sde.marginal_prob(jnp.ones_like(output), t_eps)
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous, return_state=False)
            score = score_fn(output, t_eps)
            output = output / k + batch_mul(std**2, score / k)
            output_space = to_space(output)
            output_space = merge_known_with_mask(config, output_space, known, mask, coeff)
            output = from_space(output_space)

        return inverse_scaler(output)

    return jax.pmap(langevin_projection_sampler, axis_name="batch", in_axes=(0, 0, 0, None, None))


def get_cs_solver(config, sde, model, shape, inverse_scaler, eps=1e-5):
    cs_solver = config.sampling.cs_solver
    # Probability flow ODE sampling with black-box ODE solvers
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())

    if cs_solver.lower() == "projection":
        sampling_fn = get_projection_sampler(
            config,
            sde,
            model,
            shape,
            predictor,
            corrector,
            inverse_scaler,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
        )
    elif cs_solver.lower() == "langevin":
        corrector = get_corrector("ald")
        sampling_fn = get_langevin_sampler(
            config, sde, model, shape, corrector, inverse_scaler, n_steps=config.sampling.n_steps_each, continuous=config.training.continuous, denoise=config.sampling.noise_removal, eps=eps
        )

    elif cs_solver.lower() == "langevin_projection":
        corrector = get_corrector("ald")
        sampling_fn = get_langevin_projection_sampler(
            config, sde, model, shape, corrector, inverse_scaler, n_steps=config.sampling.n_steps_each, continuous=config.training.continuous, denoise=config.sampling.noise_removal, eps=eps
        )
    elif cs_solver.lower() == "baseline":
        sampling_fn = get_baseline_sampler(
            config,
            sde,
            model,
            shape,
            predictor,
            corrector,
            inverse_scaler,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
        )
    else:
        raise ValueError(f"CS solver name {cs_solver} unknown.")

    return sampling_fn
