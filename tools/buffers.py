import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    # RolloutBufferSamples,
)
# from tools.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import GymEnv

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class RolloutBufferSamples(NamedTuple):
    states: th.Tensor
    observations: th.Tensor
    actions: th.Tensor
    last_actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    co_adv: th.Tensor
    returns: th.Tensor

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.state_space = observation_space  
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs


    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[GymEnv] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[GymEnv] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[GymEnv] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[GymEnv] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        agent_num: int=4,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.states, self.actions, self.rewards, self.advantages, self.mean_actions = None, None, None, None, None, None
        self.returns, self.episode_starts, self.last_actions, self.values, self.log_probs, self.co_adv = None, None, None, None, None, None
        self.generator_ready = False
        self.agent_num = agent_num
        self.reset()

    def reset(self) -> None:
        self.states = np.zeros((self.buffer_size, 1) + self.obs_shape, dtype=np.float32)
        self.observations = np.zeros((self.buffer_size, self.agent_num) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.agent_num + 1, self.action_dim), dtype=np.float32)
        self.last_actions = np.zeros((self.buffer_size, self.agent_num + 1, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.agent_num + 1, 1), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.agent_num + 1, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.agent_num + 1, 1), dtype=np.float32)
        self.mean_actions = np.zeros((self.buffer_size, self.agent_num + 1, 1), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.agent_num + 1, 1), dtype=np.float32)
        self.co_adv = np.zeros((self.buffer_size, self.agent_num + 1, 1), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        """
        # Convert to numpy
        # last_values = last_values.clone().cpu().numpy().flatten()
        last_values = last_values.clone().cpu().numpy()
        td_lambda = 0.5
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values      # last_values.shape=(agent_num, 1)
                # next_values = np.sum(last_values, axis=1)    # last_values.shape=(agent_num, stock_num)
                next_return = 0
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                # next_values = np.sum(self.values[step + 1], axis=1)
                next_return = self.returns[step + 1]
            # delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - np.sum(self.values[step], axis=1)            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            self.co_adv[step] = self.values[step] - self.mean_actions[step]
            self.returns[step] = td_lambda * self.gamma * next_return + \
                           (self.rewards[step] + (1 - td_lambda) * self.gamma * next_values * next_non_terminal)
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        # self.returns = self.advantages + np.sum(self.values, axis=2)

        # self.returns = self.advantages + self.values

    def add(
        self,
        states: np.ndarray,
        obs: np.ndarray,
        action: np.ndarray,
        last_action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        value_baseline: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        # if isinstance(self.observation_space, spaces.Discrete):
        #     obs = obs.reshape((self.n_envs,) + self.obs_shape)
        self.states[self.pos] = np.array(states).copy()
        self.observations[self.pos] = np.array(obs.transpose((0,2,1))).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.last_actions[self.pos] = np.array(last_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy()
        self.mean_actions[self.pos] = value_baseline.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "states",
            ]

            # for tensor in _tensor_names:
            #     self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size
        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[GymEnv] = None) -> RolloutBufferSamples:
        data = (
            self.states[batch_inds],
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.last_actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.co_adv[batch_inds],
            self.returns[batch_inds],
        )

        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


# class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    # def __init__(
    #     self,
    #     buffer_size: int,
    #     observation_space: spaces.Space,
    #     action_space: spaces.Space,
    #     device: Union[th.device, str] = "cpu",
    #     gae_lambda: float = 1,
    #     gamma: float = 0.99,
    #     n_envs: int = 1,
    # ):

    #     super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

    #     assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

    #     self.gae_lambda = gae_lambda
    #     self.gamma = gamma
    #     self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
    #     self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
    #     self.generator_ready = False
    #     self.reset()

    # def reset(self) -> None:
    #     assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
    #     self.observations = {}
    #     for key, obs_input_shape in self.obs_shape.items():
    #         self.observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)
    #     self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
    #     self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    #     self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    #     self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    #     self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    #     self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    #     self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    #     self.generator_ready = False
    #     super(RolloutBuffer, self).reset()

    # def add(
    #     self,
    #     obs: Dict[str, np.ndarray],
    #     action: np.ndarray,
    #     reward: np.ndarray,
    #     episode_start: np.ndarray,
    #     value: th.Tensor,
    #     log_prob: th.Tensor,
    # ) -> None:
    #     """
    #     :param obs: Observation
    #     :param action: Action
    #     :param reward:
    #     :param episode_start: Start of episode signal.
    #     :param value: estimated value of the current state
    #         following the current policy.
    #     :param log_prob: log probability of the action
    #         following the current policy.
    #     """
    #     if len(log_prob.shape) == 0:
    #         # Reshape 0-d tensor to avoid error
    #         log_prob = log_prob.reshape(-1, 1)

    #     for key in self.observations.keys():
    #         obs_ = np.array(obs[key]).copy()
    #         # Reshape needed when using multiple envs with discrete observations
    #         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
    #         if isinstance(self.observation_space.spaces[key], spaces.Discrete):
    #             obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
    #         self.observations[key][self.pos] = obs_

    #     self.actions[self.pos] = np.array(action).copy()
    #     self.rewards[self.pos] = np.array(reward).copy()
    #     self.episode_starts[self.pos] = np.array(episode_start).copy()
    #     self.values[self.pos] = value.clone().cpu().numpy().flatten()
    #     self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
    #     self.pos += 1
    #     if self.pos == self.buffer_size:
    #         self.full = True

    # def get(self, batch_size: Optional[int] = None) -> Generator[DictRolloutBufferSamples, None, None]:
    #     assert self.full, ""
    #     indices = np.random.permutation(self.buffer_size * self.n_envs)
    #     # Prepare the data
    #     if not self.generator_ready:

    #         for key, obs in self.observations.items():
    #             self.observations[key] = self.swap_and_flatten(obs)

    #         _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

    #         for tensor in _tensor_names:
    #             self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
    #         self.generator_ready = True

    #     # Return everything, don't create minibatches
    #     if batch_size is None:
    #         batch_size = self.buffer_size * self.n_envs

    #     start_idx = 0
    #     while start_idx < self.buffer_size * self.n_envs:
    #         yield self._get_samples(indices[start_idx : start_idx + batch_size])
    #         start_idx += batch_size

    # def _get_samples(self, batch_inds: np.ndarray, env: Optional[GymEnv] = None) -> DictRolloutBufferSamples:

    #     return DictRolloutBufferSamples(
    #         observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
    #         actions=self.to_torch(self.actions[batch_inds]),
    #         old_values=self.to_torch(self.values[batch_inds].flatten()),
    #         old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
    #         advantages=self.to_torch(self.advantages[batch_inds].flatten()),
    #         returns=self.to_torch(self.returns[batch_inds].flatten()),
    #     )