"""All functions and modules related to model definition.


这段 Python 代码似乎是与随机微分方程 (SDE) 和可能使用这些 SDE 的深度学习模型（如生成模型）相关的更大项目的一部分。
让我们通过代码并解释它的部分：
1.  import 语句：这些语句用于导入代码中需要的各种库。一些著名的库是 flax ，它用于神经网络； jax ，一个数值计算库，也允许自动微分和GPU/TPU加速；和 sde_lib ，它似乎是一个与随机微分方程 (SDE) 相关的自定义库。
2.  State 类：该类是一个数据结构，保存了训练过程中与模型状态相关的各种变量，例如优化器状态、学习率、模型状态、参数和随机数生成器状态。
3.  _MODELS 字典以及 register_model 和 get_model 函数：这些结构用于维护可在代码中使用的不同类型模型的注册表。 register_model 函数是一个装饰器，可用于将模型添加到注册表，而 get_model 可以从注册表中检索模型。
4.  get_sigmas 函数：此函数计算随机微分方程中使用的一组噪声水平（“西格玛”）。
5.  get_ddpm_params 函数：此函数检索原始去噪扩散概率模型 (DDPM) 论文中使用的参数。
6.  init_model 函数：该函数使用 flax.linen.Module 框架初始化模型。它为模型生成初始参数和状态。
7.  get_model_fn 和 get_score_fn 函数：这些是实用函数，分别为模型的前向传递和评分创建函数。
8.  to_flattened_numpy 和 from_flattened_numpy 函数：这些函数将 JAX 数组与扁平化的 numpy 数组相互转换。这有助于保存或加载模型的参数。
9.  create_classifier 函数：该函数创建一个分类器模型（特别是 WideResNet 模型），初始化其参数并可能从保存的检查点加载它们。
10. get_logit_fn 和 get_classifier_grad_fn 函数：这些函数创建与分类器相关的效用函数，例如计算分类器的 logits（pre-softmax 输出），以及分类器输出相对于其输入的梯度。
该代码似乎旨在支持使用随机微分方程训练的各种不同类型的基于分数的模型。可以使用模型注册表注册和检索模型，并提供各种实用函数来初始化模型、执行模型的前向传递、计算分数和处理模型的参数。

"""
from typing import Any

import flax
import functools
import jax.numpy as jnp
import sde_lib
import jax
import numpy as np
from models import wideresnet_noise_conditional
from flax.training import checkpoints
from utils import batch_mul
import optax


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
    step: int
    optimizer: Any
    lr: float
    model_state: Any
    ema_rate: float
    params_ema: Any
    rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = jnp.exp(jnp.linspace(jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


def init_model(rng, config):
    """Initialize a `flax.linen.Module` model."""
    model_name = config.model.name
    model_def = functools.partial(get_model(model_name), config=config)
    input_shape = (jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)
    label_shape = input_shape[:1]
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    model = model_def()
    variables = model.init({"params": params_rng, "dropout": dropout_rng}, fake_input, fake_label)
    # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
    init_model_state, initial_params = variables.pop("params")
    return model, init_model_state, initial_params


def get_model_fn(model, params, states, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: A `flax.linen.Module` object the represent the architecture of score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all mutable states.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels, rng=None):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
          rng: If present, it is the random state for dropout

        Returns:
          A tuple of (model output, new mutable states)
        """
        variables = {"params": params, **states}
        if not train:
            return model.apply(variables, x, labels, train=False, mutable=False), states
        else:
            rngs = {"dropout": rng}
            return model.apply(variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs)
            # if states:
            #   return outputs
            # else:
            #   return outputs, states

    return model_fn


def get_score_fn(sde, model, params, states, train=False, continuous=False, return_state=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
      return_state: If `True`, return the new mutable states alongside the model output.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, params, states, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t, rng=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                model, state = model_fn(x, labels, rng)
                std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                model, state = model_fn(x, labels, rng)
                std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]

            score = batch_mul(-model, 1.0 / std)
            if return_state:
                return score, state
            else:
                return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t, rng=None):
            if continuous:
                labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = jnp.round(labels).astype(jnp.int32)

            score, state = model_fn(x, labels, rng)
            if return_state:
                return score, state
            else:
                return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x):
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)


def create_classifier(prng_key, batch_size, ckpt_path):
    """Create a noise-conditional image classifier.

    Args:
      prng_key: A JAX random state.
      batch_size: The batch size of input data.
      ckpt_path: The path to stored checkpoints for this classifier.

    Returns:
      classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
      classifier_params: A dictionary that contains trainable parameters of the classifier.
    """
    input_shape = (batch_size, 32, 32, 3)
    classifier = wideresnet_noise_conditional.WideResnet(blocks_per_group=4, channel_multiplier=10, num_outputs=10)
    initial_variables = classifier.init({"params": prng_key, "dropout": jax.random.PRNGKey(0)}, jnp.ones(input_shape, dtype=jnp.float32), jnp.ones((batch_size,), dtype=jnp.float32), train=False)
    model_state, init_params = initial_variables.pop("params")
    classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)
    return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
    """Create a logit function for the classifier."""

    def preprocess(data):
        image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
        image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
        return (data - image_mean[None, ...]) / image_std[None, ...]

    def logit_fn(data, ve_noise_scale):
        """Give the logits of the classifier.

        Args:
          data: A JAX array of the input.
          ve_noise_scale: time conditioning variables in the form of VE SDEs.

        Returns:
          logits: The logits given by the noise-conditional classifier.
        """
        data = preprocess(data)
        logits = classifier.apply({"params": classifier_params}, data, ve_noise_scale, train=False, mutable=False)
        return logits

    return logit_fn


def get_classifier_grad_fn(logit_fn):
    """Create the gradient function for the classifier in use of class-conditional sampling."""

    def grad_fn(data, ve_noise_scale, labels):
        def prob_fn(data):
            logits = logit_fn(data, ve_noise_scale)
            prob = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()
            return prob

        return jax.grad(prob_fn)(data)

    return grad_fn
