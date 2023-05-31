"""All functions related to loss computation and optimization.

此 Python 代码使用 JAX 和 Flax 库提供实用函数，用于训练具有随机微分方程 (SDE) 的模型。 SDE 用于生成建模或训练基于深度能量的模型等任务。该脚本包括用于设置优化器、管理优化过程、构建损失函数的函数，以及用于训练或评估的主要步骤函数。
以下是功能细分：
get_optimizer(config) ：该函数根据提供的配置创建一个 Flax 优化器。目前仅支持 Adam 优化器。
optimization_manager(config) ：这个函数生成一个优化函数，它结合了学习率预热（在训练开始时逐渐增加学习率）和梯度裁剪（限制梯度的大小以防止不稳定的训练动态）。
get_sde_loss_fn(...) ：该函数创建了一个损失函数，用于训练具有任意 SDE 的模型。损失函数是根据去噪分数匹配 (DSM) 目标定义的。支持两种类型的分数匹配损失加权：似然加权和相关论文中推荐的加权。
该函数返回另一个函数， loss_fn ，计算给定输入的损失。
get_smld_loss_fn(...) 、 get_ddpm_loss_fn(...) ：这些函数为特定模型创建损失函数：分别是自监督多级离散化 (SMLD) 和去噪扩散概率模型 (DDPM)。
get_step_fn(...) ：该函数设置了一步训练或评估函数。根据配置，它为 SDE 选择连续或离散方法。返回的函数 step_fn 旨在与 JAX 函数 jax.lax.scan 一起使用，以高效执行多个训练步骤。
这里训练的模型似乎是基于能量的模型或使用评分函数的生成模型，例如去噪扩散隐式模型和噪声条件评分网络。值得注意的是，此处创建的损失函数与这些模型的细节有关，包括它们所基于的特定随机微分方程。
JAX 的使用表明训练过程将在硬件加速器（GPU/TPU）上运行，并且被设计为可微分的，用于基于梯度的优化方法的应用。
最后，一些评论中提到的“遗留”可能是指代码的旧版本，或用于训练此类模型的早期方法，这些模型的维护是为了比较或可重现性目的。

"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul
import optax


def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == "Adam":
        optimize_fn = optimization_manager(config)
        optimizer = optax.adamw(optimize_fn, b1=config.optim.beta1, eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported yet!")

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state, grad, warmup=config.optim.warmup, grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lr
        if warmup > 0:
            lr = lr * jnp.minimum(state.step / warmup, 1.0)
        if grad_clip >= 0:
            # Compute global gradient norm
            grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
            # Clip gradient
            clipped_grad = jax.tree_map(lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
        else:  # disabling gradient clipping if grad_clip < 0
            clipped_grad = grad
        return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

    return optimize_fn


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
        data = batch["image"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)
        rng, step_rng = random.split(rng)
        score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

        if not likelihood_weighting:
            losses = jnp.square(batch_mul(score, std) + z)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        else:
            g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
            losses = jnp.square(score + batch_mul(z, 1.0 / std))
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_smld_loss_fn(vesde, model, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        rng, step_rng = random.split(rng)
        noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
        perturbed_data = noise + data
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        target = -batch_mul(noise, 1.0 / (sigmas**2))
        losses = jnp.square(score - target)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas**2
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        losses = jnp.square(score - noise)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            params = state.optimizer.target
            states = state.model_state
            (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)
            grad = jax.lax.pmean(grad, axis_name="batch")
            new_optimizer = optimize_fn(state, grad)
            new_params_ema = jax.tree_multimap(lambda p_ema, p: p_ema * state.ema_rate + p * (1.0 - state.ema_rate), state.params_ema, new_optimizer.target)
            step = state.step + 1
            new_state = state.replace(step=step, optimizer=new_optimizer, model_state=new_model_state, params_ema=new_params_ema)
        else:
            loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch)
            new_state = state

        loss = jax.lax.pmean(loss, axis_name="batch")
        new_carry_state = (rng, new_state)
        return new_carry_state, loss

    return step_fn
