import gym

import os
import random
import time
from tqdm.auto import tqdm
import pickle as pkl
import itertools

from typing import NamedTuple

from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

class SeqReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    privileged_observations: torch.Tensor
    vision_observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    privileged_next_observations: torch.Tensor
    vision_next_observations: torch.Tensor
    cat_dones: torch.Tensor
    raw_constraints: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    p_ini_hidden_in: torch.Tensor
    p_ini_hidden_out: torch.Tensor
    mask: torch.Tensor

class SeqReplayBuffer():
    def __init__(
        self,
        buffer_size,
        observation_space,
        privileged_observation_space,
        vision_space,
        action_space,
        num_constraints,
        max_episode_length,
        seq_len,
        num_envs,
        storing_device = "cpu",
        training_device = "cpu",
        handle_timeout_termination = True,
    ):
        self.buffer_size = buffer_size
        self.max_episode_length = max_episode_length
        self.seq_len = seq_len
        self.observation_space = observation_space
        self.privileged_observation_space = privileged_observation_space.shape
        self.vision_space = vision_space
        self.action_space = action_space
        self.num_envs = num_envs

        self.num_constraints = num_constraints

        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.storing_device = storing_device
        self.training_device = training_device

        self.observations = torch.zeros((self.buffer_size, *self.observation_space), dtype=torch.float32, device = storing_device)
        self.next_observations = torch.zeros((self.buffer_size, *self.observation_space), dtype=torch.float32, device = storing_device)
        self.privileged_observations = torch.zeros((self.buffer_size, *self.privileged_observation_space), dtype=torch.float32, device = storing_device)
        self.privileged_next_observations = torch.zeros((self.buffer_size, *self.privileged_observation_space), dtype=torch.float32, device = storing_device)

        self.vision_observations = torch.zeros((self.buffer_size // 5, *self.vision_space), dtype = torch.uint8)
        self.next_vision_observations = torch.zeros_like(self.vision_observations)

        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.float32, device = storing_device)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32, device = storing_device)
        self.cat_dones = torch.zeros((self.buffer_size,), dtype=torch.float32, device = storing_device)
        self.raw_constraints = torch.zeros((self.buffer_size, self.num_constraints), dtype=torch.float32, device = storing_device)
        self.dones = torch.zeros((self.buffer_size,), dtype=torch.bool, device = storing_device)
        self.p_ini_hidden_in = torch.zeros((self.buffer_size, 1, 256), dtype=torch.float32, device = storing_device)

        # For the current episodes that started being added to the replay buffer
        # but aren't done yet. We want to still sample from them, however the masking
        # needs a termination point to not overlap to the next episode when full or even to the empty
        # part of the buffer when not full.
        self.markers = torch.zeros((self.buffer_size,), dtype=torch.bool, device = storing_device)
        self.started_adding = False

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = torch.zeros((self.buffer_size,), dtype=torch.float32, device = storing_device)

    def add(
        self,
        obs,
        privi_obs,
        vobs,
        next_obs,
        privi_next_obs,
        next_vobs,
        action,
        reward,
        cat_done,
        raw_constraints,
        done,
        p_ini_hidden_in,
        truncateds = None
    ):
        start_idx = self.pos
        stop_idx = min(self.pos + obs.shape[0], self.buffer_size)
        b_max_idx = stop_idx - start_idx

        overflow = False
        overflow_size = 0
        if self.pos + obs.shape[0] > self.buffer_size:
            overflow = True
            overflow_size = self.pos + obs.shape[0] - self.buffer_size

        assert start_idx % self.num_envs == 0, f"start_idx is not a multiple of {self.num_envs}"
        assert stop_idx % self.num_envs == 0, f"stop_idx is not a multiple of {self.num_envs}"
        assert b_max_idx == 0 or b_max_idx == self.num_envs, f"b_max_idx is not either 0 or {self.num_envs}"

        # Copy to avoid modification by reference
        self.observations[start_idx : stop_idx] = obs[: b_max_idx].clone().to(self.storing_device)
        self.vision_observations[(start_idx // (self.num_envs * 5)) * self.num_envs : (start_idx // (self.num_envs * 5)) * self.num_envs + b_max_idx] = \
            vobs[: b_max_idx].clone().to(self.storing_device)

        self.next_observations[start_idx : stop_idx] = next_obs[: b_max_idx].clone().to(self.storing_device)
        self.next_vision_observations[(start_idx // (self.num_envs * 5)) * self.num_envs : (start_idx // (self.num_envs * 5)) * self.num_envs + b_max_idx] = \
            next_vobs[: b_max_idx].clone().to(self.storing_device)

        self.privileged_observations[start_idx : stop_idx] = privi_obs[: b_max_idx].clone().to(self.storing_device)
        self.privileged_next_observations[start_idx : stop_idx] = privi_next_obs[: b_max_idx].clone().to(self.storing_device)
        self.actions[start_idx : stop_idx] = action[: b_max_idx].clone().to(self.storing_device)
        self.rewards[start_idx : stop_idx] = reward[: b_max_idx].clone().to(self.storing_device)
        self.cat_dones[start_idx : stop_idx] = cat_done[: b_max_idx].clone().to(self.storing_device)
        self.raw_constraints[start_idx : stop_idx] = raw_constraints[: b_max_idx].clone().to(self.storing_device)
        self.dones[start_idx : stop_idx] = done[: b_max_idx].clone().to(self.storing_device)
        self.p_ini_hidden_in[start_idx : stop_idx] = p_ini_hidden_in.swapaxes(0, 1)[: b_max_idx].clone().to(self.storing_device)

        # Current episodes last transition marker
        self.markers[start_idx : stop_idx] = 1
        # We need to unmark previous transitions as last
        # but only if it is not the first add to the replay buffer
        if self.started_adding:
            self.markers[self.prev_start_idx : self.prev_stop_idx] = 0
            if self.prev_overflow:
                self.markers[: self.prev_overflow_size] = 0
        self.started_adding = True
        self.prev_start_idx = start_idx
        self.prev_stop_idx = stop_idx
        self.prev_overflow = overflow
        self.prev_overflow_size = overflow_size

        if self.handle_timeout_termination:
            self.timeouts[start_idx : stop_idx] = truncateds[: b_max_idx].clone().to(self.storing_device)

        assert overflow_size == 0 or overflow_size == self.num_envs, f"overflow_size is not either 0 or {self.num_envs}"
        if overflow:
            self.full = True
            self.observations[: overflow_size] = obs[b_max_idx :].clone().to(self.storing_device)
            self.vision_observations[: overflow_size] = vobs[b_max_idx :].clone().to(self.storing_device)

            self.next_observations[: overflow_size] = next_obs[b_max_idx :].clone().to(self.storing_device)
            self.next_vision_observations[: overflow_size] = next_vobs[b_max_idx :].clone().to(self.storing_device)

            self.privileged_observations[: overflow_size] = privi_obs[b_max_idx :].clone().to(self.storing_device)
            self.privileged_next_observations[: overflow_size] = privi_next_obs[b_max_idx :].clone().to(self.storing_device)
            self.actions[: overflow_size] = action[b_max_idx :].clone().to(self.storing_device)
            self.rewards[: overflow_size] = reward[b_max_idx :].clone().to(self.storing_device)
            self.cat_dones[: overflow_size] = cat_done[b_max_idx :].clone().to(self.storing_device)
            self.raw_constraints[: overflow_size] = raw_constraints[b_max_idx :].clone().to(self.storing_device)
            self.dones[: overflow_size] = done[b_max_idx :].clone().to(self.storing_device)
            self.p_ini_hidden_in[: overflow_size] = p_ini_hidden_in.swapaxes(0, 1)[b_max_idx :].clone().to(self.storing_device)

            # Current episodes last transition marker
            self.markers[: overflow_size] = 1
            if self.handle_timeout_termination:
                self.timeouts[: overflow_size] = truncateds[b_max_idx :].clone().to(self.storing_device)
            self.pos = overflow_size
        else:
            self.pos += obs.shape[0]

    def sample(self, batch_size) -> SeqReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper_bound, size = (batch_size,), device = self.storing_device)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds) -> SeqReplayBufferSamples:
        # Using modular arithmetic we get the indices of all the transitions of the episode starting from batch_inds
        # we get "episodes" of length self.seq_len, but their true length may be less, they can have ended before that
        # we'll deal with that using a mask
        # Using flat indexing we can actually slice through a tensor using
        # different starting points for each dimension of an axis
        # as long as the slice size remains constant
        # [1, 2, 3].repeat_interleave(3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        # [1, 2, 3].repeat(3) -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
        all_indices_flat = (batch_inds.repeat_interleave(self.seq_len) + torch.arange(self.seq_len, device = self.storing_device).repeat(batch_inds.shape[0]) * self.num_envs) % self.buffer_size
        #all_indices_next_flat = (batch_inds.repeat_interleave(self.seq_len) + torch.arange(1, self.seq_len + 1, device = self.device).repeat(batch_inds.shape[0]) * self.num_envs) % self.buffer_size
        gathered_obs = self.observations[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.observations.shape[1:]))
        gathered_next_obs = self.next_observations[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.observations.shape[1:]))

        gathered_privi_obs = self.privileged_observations[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.privileged_observations.shape[1:]))
        gathered_privi_next_obs = self.privileged_next_observations[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.privileged_observations.shape[1:]))

        all_indices_flat_vision = ( (((batch_inds // self.num_envs) // 5) * self.num_envs + batch_inds % self.num_envs).repeat_interleave((self.seq_len // 5)) + \
            torch.arange(start = 0, end = (self.seq_len // 5), device = self.storing_device).repeat(batch_inds.shape[0]) * self.num_envs) % (self.buffer_size // 5)
        gathered_vobs = self.vision_observations[all_indices_flat_vision].reshape((batch_inds.shape[0], (self.seq_len // 5), *self.vision_observations.shape[1:]))
        gathered_next_vobs = self.next_vision_observations[all_indices_flat_vision].reshape((batch_inds.shape[0], (self.seq_len // 5), *self.vision_observations.shape[1:]))

        gathered_actions = self.actions[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.actions.shape[1:]))
        gathered_cat_dones = self.cat_dones[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        gathered_raw_constraints = self.raw_constraints[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, self.num_constraints))
        gathered_dones = self.dones[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        gathered_truncateds = self.timeouts[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        gathered_rewards = self.rewards[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))

        gathered_p_ini_hidden_in = self.p_ini_hidden_in[batch_inds].swapaxes(0, 1)
        gathered_p_ini_hidden_out = self.p_ini_hidden_in[(batch_inds + self.num_envs) % self.buffer_size].swapaxes(0, 1)

        gathered_markers = self.markers[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        mask = torch.cat([
            torch.ones((batch_inds.shape[0], 1), device = self.storing_device),
            (1 - (gathered_dones | gathered_markers).float()).cumprod(dim = 1)[:, 1:]
        ], dim = 1)
        data = (
            gathered_obs.to(self.training_device),
            gathered_privi_obs.to(self.training_device),
            gathered_vobs.to(self.training_device),
            gathered_actions.to(self.training_device),
            gathered_next_obs.to(self.training_device),
            gathered_privi_next_obs.to(self.training_device),
            gathered_next_vobs.to(self.training_device),
            gathered_cat_dones.to(self.training_device),
            gathered_raw_constraints.to(self.training_device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (gathered_dones.float() * (1 - gathered_truncateds)).to(self.training_device),
            gathered_rewards.to(self.training_device),
            gathered_p_ini_hidden_in.to(self.training_device),
            gathered_p_ini_hidden_out.to(self.training_device),
            mask.to(self.training_device),
        )
        return SeqReplayBufferSamples(*data)

def random_translate(imgs, pad = 4):
    n, c, h, w = imgs.size()
    imgs = torch.nn.functional.pad(imgs, (pad, pad, pad, pad)) #, mode = "replicate")
    w1 = torch.randint(0, 2*pad + 1, (n,))
    h1 = torch.randint(0, 2*pad + 1, (n,))
    cropped = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i][:] = img[:, h11:h11 + h, w11:w11 + w]
    return cropped

def random_noise(images, noise_std=0.02):
    N, C, H, W = images.size()
    noise = noise_std*torch.randn_like(images)
    noise *= torch.bernoulli(0.5*torch.ones(N, 1, 1, 1).to(images)) # Only applied on half of the images
    return images + noise

def random_shift(images, padding=4):
    N, C, H, W = images.size()
    padded_images = torch.nn.functional.pad(images, (padding, padding, padding, padding), mode='replicate')
    crop_x = torch.randint(0, 2 * padding, (N,))
    crop_y = torch.randint(0, 2 * padding, (N,))
    shifted_images = torch.zeros_like(images)
    for i in range(N):
        shifted_images[i] = padded_images[i, :, crop_y[i]:crop_y[i] + H, crop_x[i]:crop_x[i] + W]
    return shifted_images

def random_cutout(images, min_size=2, max_size=24):
    N, C, H, W = images.size()
    for i in range(N):
        size_h = random.randint(min_size, max_size)
        size_w = random.randint(min_size, max_size)
        top = random.randint(0, H - size_h)
        left = random.randint(0, W - size_w)
        coin_flip = random.random()
        if coin_flip < 0.2:
            fill_value = 0.0
            images[i, :, top:top + size_h, left:left + size_w] = fill_value
        elif coin_flip < 0.4:
            fill_value = 1.0
            images[i, :, top:top + size_h, left:left + size_w] = fill_value
        elif coin_flip < 0.6:
            fill_value = torch.rand((C, size_h, size_w), device=images.device)
            images[i, :, top:top + size_h, left:left + size_w] = fill_value
    return images

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim, seq_len, batch_size):
        super().__init__()

        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(1, 16, 5), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 4), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1568, scandots_output_dim), # 48x48
            nn.LeakyReLU(),
            nn.Linear(128, scandots_output_dim)
        )

        self.output_activation = activation

    def forward(self, vobs, hist = True, augment = False):
        bs, seql, w, h = vobs.size()

        vobs = vobs.view(-1, 1, w, h)
        if augment:
            vobs = random_cutout(random_noise(random_shift(vobs)))

        vision_latent = self.output_activation(self.image_compression(vobs))
        vision_latent = vision_latent.view(bs, seql, 128)

        if hist:
            vision_latent = vision_latent.repeat_interleave(5, axis = 1)

        return vision_latent

class ExtraCorpusQMemory(nn.Module):
    def __init__(self, seq_len, batch_size):
        super().__init__()
        self.vision = DepthOnlyFCBackbone58x87(128, seq_len, batch_size)
        self.memory = nn.GRU(128, hidden_size = 256, batch_first = True)

    def forward(self, x, hidden_in):
        if hidden_in is None:
            raise NotImplementedError

        vision_latent = self.vision(x)
        time_latent, hidden_out = self.memory(vision_latent, hidden_in)
        return time_latent, hidden_out

class QNetworkVanilla(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        self.memory = nn.GRU(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), hidden_size = 256, batch_first = True) # dummy memory for compatibility
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, a, latent):
        x = torch.cat([x, a], -1)
        x = F.elu(self.ln1(self.fc1(x)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x, None

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.memory = nn.GRU(128 + 45, hidden_size = 256, batch_first = True)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, np.prod(env.single_action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, vision_latent, hidden_in):
        if hidden_in is None:
            raise NotImplementedError

        x = torch.cat([x, vision_latent], -1)
        time_latent, hidden_out = self.memory(x, hidden_in)
        x = F.elu(self.fc1(time_latent))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = torch.tanh(self.fc_mu(x))

        return x * self.action_scale + self.action_bias, hidden_out, None

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]

def DDPG_demos_rnn_vision(cfg: DictConfig, envs):
    TOTAL_TIMESTEPS = int(cfg["train"]["params"]["config"]["total_timesteps"])
    CRITIC_LEARNING_RATE = cfg["train"]["params"]["config"]["critic_learning_rate"]
    ACTOR_LEARNING_RATE = cfg["train"]["params"]["config"]["actor_learning_rate"]
    BUFFER_SIZE = int(cfg["train"]["params"]["config"]["buffer_size"])
    LEARNING_STARTS = cfg["train"]["params"]["config"]["learning_starts"]
    GAMMA = cfg["train"]["params"]["config"]["gamma"]
    POLICY_FREQUENCY = cfg["train"]["params"]["config"]["policy_frequency"]
    TAU = cfg["train"]["params"]["config"]["tau"]
    BATCH_SIZE = cfg["train"]["params"]["config"]["batch_size"]
    POLICY_NOISE = cfg["train"]["params"]["config"]["policy_noise"]
    NOISE_CLIP = cfg["train"]["params"]["config"]["noise_clip"]
    DEMOS_RB_PATH = cfg["train"]["params"]["config"]["demos_rb_path"]
    MAX_EPISODE_LENGTH = 500
    SEQ_LEN = cfg["train"]["params"]["config"]["seq_len"]
    EVAL_AT_END = True
    NUM_CONSTRAINTS = 107
    CRITIC_NB = 10
    num_atoms = 101
    vis_h = 48
    vis_w = 48

    assert SEQ_LEN % 5 == 0, "SEQ_LEN must be a multiple of 5"
    assert BUFFER_SIZE % (5 * envs.num_envs) == 0, "BUFFER_SIZE must be a multiple of 5 * num_envs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_path = f"runs/{cfg['task']['name']}_DDPG_demos_rnn_redq_D405_z0.03_x0.24_parkour40_CORR_AUG_48x48_bigconv_seqlen{SEQ_LEN}_bs{BATCH_SIZE}_seed{cfg.seed}_{int(time.time())}"
    writer = SummaryWriter(run_path)

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    OmegaConf.save(config = cfg, f = f"{run_path}/config.yaml")

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs = ExtractObsWrapper(envs)

    rb_demos = pkl.load(open(f"{DEMOS_RB_PATH}/rb_demos.pkl", "rb"))
    ######
    rb_demos.seq_len = SEQ_LEN
    ######
    actions_min, actions_max = torch.load(f"{DEMOS_RB_PATH}/ppo_actions_minmax.pt")
    actions_min, actions_max = actions_min.to(device), actions_max.to(device)
    envs.action_space.low = actions_min.cpu().numpy()
    envs.action_space.high = actions_max.cpu().numpy()
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    vision_nn = DepthOnlyFCBackbone58x87(128, SEQ_LEN, BATCH_SIZE).to(device)
    actor = Actor(envs).to(device)

    QNetwork = QNetworkVanilla
    qfs = [QNetwork(envs, device = device).to(device) for _ in range(CRITIC_NB)]
    qf_targets = [QNetwork(envs, device = device).to(device) for _ in range(CRITIC_NB)]
    for i in range(CRITIC_NB):
        qf_targets[i].load_state_dict(qfs[i].state_dict())

    q_optimizer = optim.Adam(itertools.chain(*([q.parameters() for q in qfs])), lr=CRITIC_LEARNING_RATE)
    qf1 = QNetwork(envs, device = device).to(device)
    qf2 = QNetwork(envs, device = device).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()) + list(vision_nn.parameters()), lr=ACTOR_LEARNING_RATE)

    envs.single_observation_space.dtype = np.float32
    rb = SeqReplayBuffer(
        BUFFER_SIZE,
        (45,),
        envs.single_observation_space,
        (vis_h, vis_w),
        envs.single_action_space,
        NUM_CONSTRAINTS,
        MAX_EPISODE_LENGTH,
        SEQ_LEN,
        envs.num_envs,
        storing_device = "cpu",
        training_device = device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs_privi = envs.reset()
    obs = obs_privi.clone()[:, : 45]
    vobs = torch.zeros((envs.num_envs, vis_h, vis_w), device = device)
    next_vobs = vobs.clone()
    vision_latent = None

    isaac_env_steps = TOTAL_TIMESTEPS // envs.num_envs
    print(f"Starting training for {isaac_env_steps} isaac env steps")

    gru_p_hidden_in = torch.zeros((actor.memory.num_layers, envs.num_envs, actor.memory.hidden_size), device = device) # p for policy
    gru_p_hidden_out = gru_p_hidden_in.clone()

    true_steps_num = 0
    actor_training_step = 0
    for global_step in range(isaac_env_steps):
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            if global_step % 5 == 0:
                vision_latent = vision_nn(vobs.unsqueeze(1), hist = False)
        if true_steps_num < LEARNING_STARTS:
            actions = torch.zeros((envs.num_envs, rb.action_dim), device = device).uniform_(-1, 1)
        else:
            with torch.no_grad():
                actions, gru_p_hidden_out, _ = actor(obs.unsqueeze(1), vision_latent, gru_p_hidden_in)
                actions = actions.squeeze(1)

        true_steps_num += envs.num_envs

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs_privi, rewards, terminations, infos = envs.step(actions)
        true_dones = infos["true_dones"]
        truncateds = infos["truncateds"].float()
        raw_constraints = infos["raw_constraints"]

        next_obs = next_obs_privi.clone()[:, : 45]

        real_next_obs_privi = next_obs_privi.clone()
        real_next_obs = next_obs.clone()
        if(true_dones.any()):
            true_dones_idx = torch.argwhere(true_dones).squeeze()
            gru_p_hidden_out[:, true_dones_idx] = 0

        if "depth" in infos:
            next_vobs = infos["depth"].clone()[..., 19:-18]

        rb.add(obs, obs_privi, (vobs * 255).to(torch.uint8), real_next_obs, real_next_obs_privi, (next_vobs * 255).to(torch.uint8), \
            actions, rewards, terminations, raw_constraints, true_dones, gru_p_hidden_in, truncateds)
        gru_p_hidden_in = gru_p_hidden_out

        if (global_step + 1) % 24 == 0:
            for el, v in zip(list(envs.episode_sums.keys())[:envs.numRewards], (torch.mean(envs.rew_mean_reset, dim=0)).tolist()):
                writer.add_scalar(f"reward/{el}", v, (global_step + 1) // 24)
            writer.add_scalar(f"reward/cum_rew", (torch.sum(torch.mean(envs.rew_cum_reset, dim=0))).item(), (global_step + 1) // 24)
            writer.add_scalar(f"reward/avg_rew", envs.terrain_levels.float().mean().item(), (global_step + 1) // 24)
            for el, v in zip(envs.cstr_manager.get_names(), (100.0 * torch.mean(envs.cstr_mean_reset, dim=0)).tolist()):
                writer.add_scalar(f"cstr/{el}", v, (global_step + 1) // 24)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs_privi = next_obs_privi
        obs = next_obs
        vobs = next_vobs

        # ALGO LOGIC: training.
        if true_steps_num > LEARNING_STARTS:
            for local_steps in range(8):
                data = rb.sample(BATCH_SIZE // 2)
                data_expert = rb_demos.sample(BATCH_SIZE // 2)
                data = SeqReplayBufferSamples(
                    observations=torch.cat([data.observations, data_expert.observations], dim=0),
                    privileged_observations=torch.cat([data.privileged_observations, data_expert.privileged_observations], dim=0),
                    vision_observations=torch.cat([data.vision_observations, data_expert.vision_observations], dim=0).float() / 255,
                    actions=torch.cat([data.actions, data_expert.actions], dim=0),
                    next_observations=torch.cat([data.next_observations, data_expert.next_observations], dim=0),
                    privileged_next_observations=torch.cat([data.privileged_next_observations, data_expert.privileged_next_observations], dim=0),
                    vision_next_observations=torch.cat([data.vision_next_observations, data_expert.vision_next_observations], dim=0).float() / 255,
                    cat_dones=torch.cat([data.cat_dones, data_expert.cat_dones], dim=0),
                    raw_constraints=torch.cat([data.raw_constraints, data_expert.raw_constraints], dim=0),
                    dones=torch.cat([data.dones, data_expert.dones], dim=0),
                    rewards=torch.cat([data.rewards, data_expert.rewards], dim=0),
                    p_ini_hidden_in=torch.cat([data.p_ini_hidden_in, data_expert.p_ini_hidden_in], dim=1),
                    p_ini_hidden_out=torch.cat([data.p_ini_hidden_out, data_expert.p_ini_hidden_out], dim=1),
                    mask=torch.cat([data.mask, data_expert.mask], dim = 0),
                )

                norm_const = data.raw_constraints / envs.cstr_manager.get_running_maxes()
                recomputed_cat_dones = (envs.cstr_manager.min_p + torch.clamp(
                    norm_const,
                    min=0.0,
                    max=1.0
                ) * (envs.cstr_manager.get_max_p() - envs.cstr_manager.min_p)).max(-1).values

                with torch.no_grad():
                    clipped_noise = (torch.randn_like(data.actions, device=device) * POLICY_NOISE).clamp(
                        -NOISE_CLIP, NOISE_CLIP
                    ) #* actor.action_scale

                    vlatent = vision_nn(data.vision_next_observations)
                    qvlatent = None
                    next_state_actions, _, _ = actor(data.next_observations, vlatent, data.p_ini_hidden_out)
                    next_state_actions = (next_state_actions + clipped_noise).clamp(
                        actions_min, actions_max
                    )
                    targets_selected = torch.randperm(CRITIC_NB)[:2]
                    qf_next_targets = torch.stack([qf_targets[i](data.privileged_next_observations, next_state_actions, qvlatent)[0] for i in targets_selected])
                    min_qf_next_target = qf_next_targets.min(dim = 0).values
                    next_q_value = (1 - recomputed_cat_dones.flatten()) * data.rewards.flatten() + (1 - recomputed_cat_dones.flatten()) * (1 - data.dones.flatten()) * GAMMA * (min_qf_next_target).view(-1)

                true_samples_nb = data.mask.sum()
                qvlatent = None
                qf_a_values = torch.stack([qf(data.privileged_observations, data.actions, qvlatent)[0].view(-1) for qf in qfs])
                qf_loss = (((qf_a_values - next_q_value.unsqueeze(0)) ** 2) * data.mask.view(-1).unsqueeze(0)).sum() / (true_samples_nb * CRITIC_NB)

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                nn.utils.clip_grad_norm_(qf1.parameters(), 0.5)
                q_optimizer.step()

                if local_steps % POLICY_FREQUENCY == 0:
                    for i in range(CRITIC_NB):
                        for param, target_param in zip(qfs[i].parameters(), qf_targets[i].parameters()):
                            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                if local_steps == 7:
                    actor_training_step += 1
                    vlatent = vision_nn(data.vision_observations, augment = True)
                    qvlatent = None
                    diff_actions, _, recons_height_maps = actor(data.observations, vlatent, data.p_ini_hidden_in)

                    qs = torch.stack([qf(data.privileged_observations, diff_actions, qvlatent)[0] for qf in qfs])
                    actor_loss = -(qs.squeeze(-1) * data.mask.unsqueeze(0)).sum() / (true_samples_nb * CRITIC_NB)
                    final_loss = actor_loss

                    actor_optimizer.zero_grad()
                    final_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()

                    writer.add_scalar(f"loss/actor_loss", actor_loss.item(), actor_training_step)
                    writer.add_scalar(f"infos/Q_max", qs.max().item(), actor_training_step)

        if (global_step + 1) % (500 * 24) == 0:
            model_path = f"{run_path}/cleanrl_model_e{(global_step + 1) // 24}.pt"
            torch.save((actor.state_dict(), vision_nn.state_dict()), model_path)
            print("Saved model checkpoint")

        if (global_step + 1) % (5 * 24) == 0:
            model_path = f"{run_path}/cleanrl_model.pt"
            torch.save((actor.state_dict(), vision_nn.state_dict()), model_path)
            print("Saved model")

def eval_DDPG_demos_rnn_vision(cfg: DictConfig, envs):
    import cv2
    vis_h = 48
    vis_w = 48
    is_video_gen = "video_save_path" in cfg["task"]

    if is_video_gen:
        video_save_path = cfg["task"]["video_save_path"]
        position = video_save_path.rfind(".mp4")
        output_file = video_save_path[:position] + "_depth" + video_save_path[position:]
        fps = 10

    checkpoint = cfg["checkpoint"]
    actor_sd, vision_sd = torch.load(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs = ExtractObsWrapper(envs)

    vision_nn = DepthOnlyFCBackbone58x87(128, 0, 0).to(device)
    vision_nn.load_state_dict(vision_sd)
    actor = Actor(envs).to(device)
    actor.load_state_dict(actor_sd)

    obs_privi = envs.reset()
    obs = obs_privi.clone()[:, : 45]
    vobs = torch.zeros((envs.num_envs, vis_h, vis_w), device = device)
    next_vobs = vobs.clone()
    vision_latent = None
    gru_p_hidden_in = torch.zeros((actor.memory.num_layers, envs.num_envs, actor.memory.hidden_size), device = device) # p for policy

    if is_video_gen:
        depth_images = []

    for global_step in range(2000):
        with torch.no_grad():
            if global_step % 5 == 0:
                vision_latent = vision_nn(vobs.unsqueeze(1), hist = False)

        with torch.no_grad():
            actions, gru_p_hidden_out, _ = actor(obs.unsqueeze(1), vision_latent, gru_p_hidden_in)
            actions = actions.squeeze(1)

        next_obs_privi, rewards, terminations, infos = envs.step(actions)
        next_obs = next_obs_privi.clone()[:, : 45]
        obs = next_obs
        if "depth" in infos:
            next_vobs = infos["depth"].clone()[..., 19:-18]
        vobs = next_vobs
        gru_p_hidden_in = gru_p_hidden_out

        if is_video_gen:
            depth_images.append((vobs * 255).to(torch.uint8).squeeze().cpu().numpy())
            if global_step == 200:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, fps, (vis_w, vis_h), isColor=True)
                out.set(cv2.VIDEOWRITER_PROP_QUALITY, 1)  # 1 is the highest quality setting
                for i in range(len(depth_images)):
                    # Get the i-th frame
                    frame = depth_images[i]

                    if frame.shape != (vis_h, vis_w):
                        raise ValueError(f"Frame {i} has incorrect shape: {frame.shape}")

                    # Write the frame to the video (as a single-channel image)
                    rgb_image = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)
                    out.write(rgb_image)

                # Release the VideoWriter object
                out.release()
                print(f"Video saved to {output_file}")
