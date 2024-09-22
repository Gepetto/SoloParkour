import gym

import os
import random
import time
from tqdm.auto import tqdm
import pickle as pkl

from typing import NamedTuple

from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

class RunningMeanStd(nn.Module):
    def __init__(self, shape = (), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update = True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = update_mean_var_count_from_moments(
            self.running_mean, self.running_var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        self.obs_rms = RunningMeanStd(shape = envs.single_observation_space.shape)
        self.value_rms = RunningMeanStd(shape = ())

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        return action_mean, None, None, None

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
        critic_rnn = False,
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
        self.critic_rnn = critic_rnn
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

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]

def DDPG_demos_generate(cfg: DictConfig, envs):
    TARGET_POLICY_PATH = cfg["train"]["params"]["config"]["target_policy_path"]
    DEMOS_BUFFER_SIZE = int(cfg["train"]["params"]["config"]["demos_buffer_size"])
    MAX_EPISODE_LENGTH = 500
    SEQ_LEN = 5
    NUM_CONSTRAINTS = 107
    RNN_CRITIC = True
    vis_h = 48
    vis_w = 48

    assert SEQ_LEN % 5 == 0, "SEQ_LEN must be a multiple of 5"
    assert DEMOS_BUFFER_SIZE % (5 * envs.num_envs) == 0, "DEMO_BUFFER_SIZE must be a multiple of 5 * num_envs"

    actor_sd = torch.load(TARGET_POLICY_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_path = f"runs/{cfg['task']['name']}_DDPG_demos_generate_D405_parkour40_CORR_CONTROL_{int(time.time())}"
    writer = SummaryWriter(run_path)

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    OmegaConf.save(config = cfg, f = f"{run_path}/config.yaml")

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs = ExtractObsWrapper(envs)

    actor_demos = Agent(envs).to(device)
    actor_demos.load_state_dict(actor_sd)
    print("Target actor for demonstrations loaded succesfully.")
    envs.single_observation_space.dtype = np.float32
    rb_demos = SeqReplayBuffer(
        DEMOS_BUFFER_SIZE,
        (45,),
        envs.single_observation_space,
        (vis_h, vis_w),
        envs.single_action_space,
        NUM_CONSTRAINTS,
        MAX_EPISODE_LENGTH,
        SEQ_LEN,
        envs.num_envs,
        critic_rnn = RNN_CRITIC,
        storing_device = "cpu",
        training_device = device,
        handle_timeout_termination=True,
    )

    filling_iterations = DEMOS_BUFFER_SIZE // envs.num_envs
    obs_privi = envs.reset()
    obs = obs_privi.clone()[:, : 45]
    vobs = torch.zeros((envs.num_envs, vis_h, vis_w), dtype = torch.uint8, device = device)
    next_vobs = vobs.clone()
    with torch.no_grad():
        dummy_actions, _, _, _ = actor_demos.get_action_and_value(actor_demos.obs_rms(obs_privi, update = False))
        actions_min, _ = dummy_actions.min(dim = 0)
        actions_max, _ = dummy_actions.max(dim = 0)

    dummy_hidddens = torch.zeros((1, envs.num_envs, 256), device = device)

    for fi in range(filling_iterations):
        with torch.no_grad():
            actions, _, _, _ = actor_demos.get_action_and_value(actor_demos.obs_rms(obs_privi, update = False))
            cur_min, _ = actions.min(dim = 0)
            cur_max, _ = actions.max(dim = 0)
            actions_min = torch.min(actions_min, cur_min)
            actions_max = torch.max(actions_max, cur_max)

        next_obs_privi, rewards, terminations, infos = envs.step(actions)
        true_dones = infos["true_dones"].float()
        truncateds = infos["truncateds"].float()
        raw_constraints = infos["raw_constraints"]

        next_obs = next_obs_privi.clone()[:, : 45]

        real_next_obs_privi = next_obs_privi.clone()
        real_next_obs = next_obs.clone()

        if "depth" in infos:
            next_vobs = (infos["depth"].clone()[..., 19:-18] * 255).to(torch.uint8)

        rb_demos.add(obs, obs_privi, vobs, real_next_obs, real_next_obs_privi, next_vobs, actions, rewards, terminations, raw_constraints, true_dones, dummy_hidddens, truncateds)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs_privi = next_obs_privi
        obs = next_obs
        vobs = next_vobs

    torch.save((actions_min.cpu(), actions_max.cpu()), f"{run_path}/ppo_actions_minmax.pt")
    pkl.dump(rb_demos, open(f"{run_path}/rb_demos.pkl", "wb"))
    print(f"Demo replay buffer filled with {DEMOS_BUFFER_SIZE} experts demonstrations")
