"""ORBiT (object-centric transformer) for JAXtari.

JAX port of the ORBiT agent from OCRL_Transformer (``cleanrl_experiments/cleanrl/ppo_atari_orbit.py``)
and its OARC ``EgoCentricWrapper`` (taken from ORBiT_JAX), here ``ORCAWrapper`` with a configurable anchor.

Usage: object-centric runs (PIXEL_BASED: False) of PPO and PQN
    uv run main.py +alg=pqn_oc_paper ORBIT=True     # ORCA observation, ORBiT transformer
    uv run main.py +alg=ppo_oc_tuned ORCA=True      # ORCA observation, the usual MLP

ORCA observation (ORCAWrapper), in place of ObjectCentricWrapper's stacked object vector:
one (max_len, num_object_types + 4) matrix per step, a row per object slot,

    row = (one-hot category, dx, dy, center_x, center_y)

with positions and velocities relative to an anchor object category (ORCA_ANCHOR, "auto" = the
game's player), present objects packed at the top and zero rows below. There is no frame stack: the
rows carry the velocities. For the MLP it is normalised (by the observation space's bounds) and
flattened, like the default object-centric observation; ORBiT gets the raw matrix.

ORBiT network: a transformer over the rows (the padding mask comes from the one-hot columns),
followed by pooling; PPO puts its actor and critic heads on top, PQN a Q head.

The model reproduces the PyTorch original numerically (checked against it in ORBiT_JAX):
PyTorch's default initialisations, identical initial weights in all transformer blocks (use
``init_orbit``, as torch's TransformerEncoder deep-copies one layer), and mean pooling that sums
padded positions too but divides by the number of real objects.
"""
import functools
import math
from typing import Any, Dict, Literal, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import unfreeze
from flax.linen.initializers import constant, orthogonal, zeros

import jaxtari.spaces as spaces
from jaxtari.environment import ObjectObservation
from jaxtari.wrappers import (
    AtariState,
    JaxtariWrapper,
    _bcast_where,
    _find_atari_wrapper,
    _reset_with_active,
    _unroll_frame_skip,
    speculative_cond,
)


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

# ORCA_ANCHOR: auto -> the player category, for the games that don't call it "player"
AUTO_ANCHORS = {"freeway": "chicken", "frostbite": "bailey", "gravitar": "ship", "skiing": "skier"}


def orca_settings(config: dict) -> Optional[dict]:
    """ORCA observation settings for an agent's make_env from the run config, or None.

    ORCA (MLP) and ORBIT (transformer, implies ORCA) apply to object-centric runs only.
    `flatten` is True for the MLP, which gets the observation normalised and flattened.
    """
    orbit = bool(config.get("ORBIT", False))
    if not (orbit or config.get("ORCA", False)) or config.get("PIXEL_BASED", True):
        return None
    if config.get("REWARD_MACHINE", False):
        raise ValueError("ORCA/ORBIT and REWARD_MACHINE cannot be combined (the RM state is appended to a flat observation)")
    anchor = config.get("ORCA_ANCHOR", "auto") or ""
    if anchor == "auto":
        anchor = AUTO_ANCHORS.get(str(config["ENV_ID"]).lower(), "player")
    return {"anchor": anchor, "include_wh": bool(config.get("ORCA_INCLUDE_WH", False)), "flatten": not orbit}


def wrap_orca(env, orca: dict, eval: bool = False):
    """AtariWrapper -> ORCA observation (normalised and flattened for the MLP)."""
    from jaxtari.wrappers import FlattenObservationWrapper, NormalizeObservationWrapper

    env = ORCAWrapper(env, anchor=orca["anchor"], include_wh=orca["include_wh"],
                            frame_skip=4, clip_reward=not eval)
    if orca["flatten"]:
        env = FlattenObservationWrapper(NormalizeObservationWrapper(env))
    return env


def uses_orbit(orca: Optional[dict]) -> bool:
    """True if the run uses the ORBiT network (ORBIT), False for the MLP (ORCA alone or default)."""
    return orca is not None and not orca["flatten"]


def orbit_model_kwargs(config: dict) -> dict:
    """Transformer hyperparameters of ORBiTNetwork / ORBiTQNetwork from the run config."""
    return {"emb_dim": int(config.get("ORBIT_EMB_DIM", 128)),
            "num_heads": int(config.get("ORBIT_NUM_HEADS", 8)),
            "num_blocks": int(config.get("ORBIT_NUM_BLOCKS", 3)),
            "pooling_type": config.get("ORBIT_POOLING", "mean"),
            "num_post_layers": int(config.get("ORBIT_NUM_POST_LAYERS", 0))}


def orca_run_suffix(config: dict) -> str:
    """'_orbit' / '_orca' for run names, '' with the default observation."""
    orca = orca_settings(config)
    if orca is None:
        return ""
    return "_orbit" if uses_orbit(orca) else "_orca"


# --------------------------------------------------------------------------------------------
# Observation wrapper
# --------------------------------------------------------------------------------------------

def _is_obj(x):
    return isinstance(x, ObjectObservation)


def _path_name(path) -> str:
    key = path[-1]
    for attr in ("name", "key", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


@struct.dataclass
class ORCAState:
    """`atari_state` is the name run_utils.game_states walks to reach the env state."""
    atari_state: AtariState
    prev_x: jax.Array       # (num_slots,) top-left x of every slot at the previous agent step
    prev_y: jax.Array
    prev_active: jax.Array  # (num_slots,) bool


class ORCAWrapper(JaxtariWrapper):
    """ORCA: object-centric, anchor-relative observation of shape ``(max_len, brutto_feature_size)``.

    The anchor is an object category of the game (``anchor``, e.g. the player); positions and
    velocities are relative to its first present object, or absolute with ``anchor=""`` or while
    no anchor object is present.

    Apply after ``AtariWrapper``, in place of ObjectCentricWrapper. Frame skip, reward summing,
    reward clipping and auto-reset on game over match ObjectCentricWrapper; there is no frame
    stack (the rows carry velocities) and no observation normalisation (as in the original).

    Velocities are the displacement of an object's top-left corner since the previous agent step,
    as in OCAtari (``o.dx``, ``o.dy``); objects that just (re)appeared have zero velocity.
    """

    def __init__(self, env, anchor: str = "player",
                 type_embedding: Optional[Literal["one_hot"]] = "one_hot",
                 include_wh: bool = False, frame_skip: int = 4, clip_reward: bool = True,
                 exclude_classes: Tuple[str, ...] = ()):
        super().__init__(env)
        atari_wrapper = _find_atari_wrapper(env)
        assert atari_wrapper is not None, "ORCAWrapper must be applied after AtariWrapper"
        self._speculative_autoreset = bool(atari_wrapper.noop_reset or atari_wrapper.first_fire)
        self.frame_skip = frame_skip
        self.include_wh = include_wh
        self.clip_reward = clip_reward

        # Discover the object categories (= ObjectObservation leaves) and their number of slots
        obs_shape, _ = jax.eval_shape(self._env.reset, jax.random.PRNGKey(0))
        leaves = jax.tree_util.tree_flatten_with_path(obs_shape, is_leaf=_is_obj)[0]
        categories = [(_path_name(p), int(np.prod(o.x.shape))) for p, o in leaves if _is_obj(o)]
        if not categories:
            raise ValueError("The environment's observation contains no ObjectObservation.")
        self.categories = [c for c, _ in categories]
        self.max_objs = dict(categories)
        self.num_object_types = len(categories)
        self.max_len = sum(n for _, n in categories)

        # Slot -> category index, and which slots are dropped (exclude_classes) or belong to the anchor
        type_ids = np.concatenate([np.full(n, i) for i, (_, n) in enumerate(categories)])
        lower = [c.lower() for c in self.categories]
        self.exclude = np.array([lower[t] in {e.lower() for e in exclude_classes} for t in type_ids])
        self.anchor = anchor
        if anchor:
            if anchor.lower() not in lower:
                raise ValueError(f"Anchor category '{anchor}' not found, available: {self.categories} "
                                 f"(set ORCA_ANCHOR, or ORCA_ANCHOR='' for absolute coordinates)")
            self.is_anchor = type_ids == lower.index(anchor.lower())
        else:
            self.is_anchor = np.zeros_like(type_ids, dtype=bool)
        self.type_ids = type_ids

        self.netto_feature_size = 6 if include_wh else 4
        self.brutto_feature_size = self.netto_feature_size
        if type_embedding == "one_hot":
            self.brutto_feature_size += self.num_object_types
            self.type_emb = np.eye(self.num_object_types, self.brutto_feature_size, dtype=np.float32)[type_ids]
        elif type_embedding is None:
            self.type_emb = np.zeros((self.max_len, self.brutto_feature_size), dtype=np.float32)
        else:
            raise AttributeError(f"Type embedding {type_embedding} not supported!")

        # finite per-feature bounds, for NormalizeObservationWrapper (ORCA with the MLP):
        # one-hot in [0, 1], width/height in [0, screen], centers relative to the anchor in
        # [-screen, screen], velocities (an object's minus the anchor's) in [-2 screen, 2 screen]
        img_h, img_w = self._env.image_space().shape[:2]
        low = [0.0] * (self.brutto_feature_size - self.netto_feature_size)
        high = [1.0] * (self.brutto_feature_size - self.netto_feature_size)
        if include_wh:
            low, high = low + [0.0, 0.0], high + [img_w, img_h]
        low, high = low + [-2 * img_w, -2 * img_h, -img_w, -img_h], high + [2 * img_w, 2 * img_h, img_w, img_h]
        shape = (self.max_len, self.brutto_feature_size)
        self._observation_space = spaces.Box(np.broadcast_to(np.array(low, np.float32), shape),
                                             np.broadcast_to(np.array(high, np.float32), shape), shape=shape)

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _slots(self, obs):
        objs = [o for o in jax.tree.leaves(obs, is_leaf=_is_obj) if _is_obj(o)]
        cat = lambda f: jnp.concatenate([jnp.ravel(getattr(o, f)).astype(jnp.float32) for o in objs])
        x, y, w, h = cat("x"), cat("y"), cat("width"), cat("height")
        return x, y, w, h, cat("active") > 0

    def _orca_obs(self, obs, prev_x, prev_y, prev_active):
        x, y, w, h, active = self._slots(obs)
        moved = active & prev_active
        dx, dy = jnp.where(moved, x - prev_x, 0.0), jnp.where(moved, y - prev_y, 0.0)
        cx, cy = x + w / 2, y + h / 2

        pv = jnp.stack([dx, dy, cx, cy], axis=-1)
        feats = jnp.concatenate([jnp.stack([w, h], axis=-1), pv], axis=-1) if self.include_wh else pv

        # relative position and velocity w.r.t. the first present anchor object (if any),
        # found before excluded classes are dropped, as in the original
        is_anchor = active & self.is_anchor
        has_anchor = is_anchor.any()
        anchor_pv = pv[jnp.argmax(is_anchor)]
        rel = jnp.where(has_anchor, anchor_pv, 0.0)
        feats = feats.at[:, -4:].add(-rel)

        state = jnp.concatenate([jnp.zeros((self.max_len, self.brutto_feature_size - self.netto_feature_size)),
                                 feats], axis=-1) + self.type_emb
        keep = active & ~self.exclude
        state = jnp.where(keep[:, None], state, 0.0)

        # pack the kept objects at the top, keeping their order (as OCAtari's object list does)
        order = jnp.argsort(jnp.where(keep, 0, 1), stable=True)
        return state[order], (x, y, active)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[jax.Array, ORCAState]:
        return self._reset_active(key, True)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_active(self, key: chex.PRNGKey, active: chex.Array = True) -> Tuple[jax.Array, ORCAState]:
        # `active` = False lanes of a batched auto-reset are discarded, so they skip the no-ops
        obs, atari_state = _reset_with_active(self._env, key, active)
        x, y, _, _, act = self._slots(obs)
        orca_obs, _ = self._orca_obs(obs, x, y, act)
        return orca_obs, ORCAState(atari_state, x, y, act)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: ORCAState, action: int
             ) -> Tuple[jax.Array, ORCAState, float, bool, bool, Dict[Any, Any]]:
        atari_state, latest_obs, _, rewards, terminations, truncations, info = _unroll_frame_skip(
            self._env, state.atari_state, action, self.frame_skip)

        orca_obs, (x, y, active) = self._orca_obs(latest_obs, state.prev_x, state.prev_y, state.prev_active)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()

        # auto-reset on game over (not on a lost life) or truncation, as ObjectCentricWrapper:
        # under vmap a plain lax.cond runs the whole reset (with its no-ops) for every env on
        # every step; speculative_cond only runs it when some env is done
        done = jnp.logical_or(info["env_done"], truncated)
        step_state = ORCAState(atari_state, x, y, active)
        if self._speculative_autoreset:
            orca_obs, new_state = speculative_cond(
                lambda done, orca_obs, state: done,
                lambda done, orca_obs, state: self._reset_active(state.atari_state.key, done),
                lambda done, orca_obs, state: (orca_obs, state),
            )(done, orca_obs, step_state)
        else:
            orca_obs, new_state = _bcast_where(done, self._reset_active(atari_state.key, done), (orca_obs, step_state))
        return orca_obs, new_state, reward, terminated, truncated, info


# --------------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------------

def _uniform(bound):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


class TorchDense(nn.Module):
    """``nn.Dense`` with ``torch.nn.Linear``'s default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias."""
    features: int
    bias_init_zero: bool = False

    @nn.compact
    def __call__(self, x):
        bound = 1 / math.sqrt(x.shape[-1])
        return nn.Dense(self.features, kernel_init=_uniform(bound),
                        bias_init=zeros if self.bias_init_zero else _uniform(bound))(x)


class MultiheadSelfAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` (self-attention, batch_first) with ``key_padding_mask``."""
    emb_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask):
        E, H = self.emb_dim, self.num_heads
        D = E // H
        # torch: xavier_uniform_ over the stacked (3E, E) in_proj_weight, zero in_proj_bias
        in_init = _uniform(math.sqrt(6 / (E + 3 * E)))
        q, k, v = (nn.Dense(E, kernel_init=in_init, bias_init=zeros, name=n)(x) for n in ("q_proj", "k_proj", "v_proj"))
        split = lambda t: t.reshape(t.shape[:-1] + (H, D))
        q, k, v = split(q), split(k), split(v)

        logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / math.sqrt(D)
        logits = jnp.where(mask[..., None, None, :], logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("...hqk,...khd->...qhd", weights, v).reshape(x.shape[:-1] + (E,))
        return TorchDense(E, bias_init_zero=True, name="out_proj")(out)


class TransformerEncoderLayer(nn.Module):
    """``torch.nn.TransformerEncoderLayer`` with ``norm_first=False``, ``activation=relu``, no dropout."""
    emb_dim: int
    num_heads: int
    dim_feedforward: int

    @nn.compact
    def __call__(self, x, mask):
        a = MultiheadSelfAttention(self.emb_dim, self.num_heads, name="self_attn")(x, mask)
        x = nn.LayerNorm(epsilon=1e-5, name="norm1")(x + a)
        f = TorchDense(self.emb_dim, name="linear2")(nn.relu(TorchDense(self.dim_feedforward, name="linear1")(x)))
        return nn.LayerNorm(epsilon=1e-5, name="norm2")(x + f)


class ORBiTNetwork(nn.Module):
    """ORBiT torso: object encoder, transformer blocks, optional post layers and pooling.

    Input ``(..., max_len, features)`` from ORCAWrapper, output ``(..., emb_dim)``.
    Used as PPO's ``Network``, under the usual Actor and Critic heads.
    """
    num_object_types: int
    emb_dim: int = 128
    num_heads: int = 8
    num_blocks: int = 3
    pooling_type: str = "mean"
    num_post_layers: int = 0

    def mask(self, x):
        mask = jnp.any(x[..., :self.num_object_types] != 0, axis=-1)
        return mask | ~mask.any(-1, keepdims=True)  # if there is no object, attend over all positions

    def pooling(self, x, mask):
        if self.pooling_type == "max":
            return jnp.max(x * mask[..., None], axis=-2)
        if self.pooling_type == "mean":
            # as in the original: padded positions are summed too, only the count is masked
            return jnp.sum(x, axis=-2) / jnp.sum(mask, axis=-1, keepdims=True)
        if self.pooling_type == "first":
            return x[..., 0, :]
        raise NotImplementedError(self.pooling_type)

    def features(self, x):
        mask = self.mask(x)
        h = TorchDense(self.emb_dim, name="encoder")(x)
        for i in range(self.num_blocks):
            h = TransformerEncoderLayer(self.emb_dim, self.num_heads, self.emb_dim, name=f"block_{i}")(h, mask)
        for i in range(self.num_post_layers):
            h = nn.relu(TorchDense(self.emb_dim, name=f"post_{i}")(h))
        return self.pooling(h, mask)

    @nn.compact
    def __call__(self, x):
        return self.features(x)


class ORBiTQNetwork(ORBiTNetwork):
    """The ORBiT torso with a Q-value head (PQN)."""
    action_dim: int = 0

    @nn.compact
    def __call__(self, x):
        hidden = self.features(x)
        # head init as MLP_QNetwork's
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="q")(hidden)


def init_orbit(network: ORBiTNetwork, key, sample_obs):
    """``network.init``, then every transformer block starts from block_0's weights, as in
    ``torch.nn.TransformerEncoder`` (which deep-copies one layer)."""
    params = unfreeze(network.init(key, sample_obs))
    for i in range(1, network.num_blocks):
        params["params"][f"block_{i}"] = jax.tree.map(jnp.copy, params["params"]["block_0"])
    return params
