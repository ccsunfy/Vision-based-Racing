import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union

from stable_baselines3.common import logger
import os, sys
from gymnasium import spaces

import torch as th
from utils.policies.td_policies import TD3Policy, CnnPolicy, BasePolicy, MultiInputPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn
from tqdm import tqdm
from  stable_baselines3.common.utils import polyak_update, get_parameters_by_name

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


class TemporalDifferBase:
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
        "CnnPolicy": CnnPolicy,
        "TD3Policy": TD3Policy,
    }
    observation_space: spaces.Space
    action_space: spaces.Space
    num_envs: int
    lr_schedule: Schedule

    def __init__(
            self,
            env,
            policy: Union[Type, str],
            policy_kwargs: Optional[Dict] = None,
            learning_rate: Union[float, Schedule] = 1e-3,
            logger_kwargs: Optional[Dict[str, Any]] = None,
            commit: Optional[str] = None,
            save_path: Optional[str] = sys,
            dump_step: int = 1e4,
            horizon: float = 1,
            tau: float = 0.005,
            gamma: float = 0.95,
            lamda: float = 0.95,
            policy_noise: float = 0.,
            device: Optional[str] = "cpu",

            **kwargs
    ):
        root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.save_path = f"{root}/saved" if save_path is None else save_path
        self.device = th.device(device)

        self.env = env
        self.env.to(self.device)
        self.num_envs = env.num_envs
        self.observation_space: spaces.Dict = env.observation_space
        self.action_space: spaces.Box = env.action_space

        self._dump_step = dump_step
        self.learning_rate = learning_rate
        self.commit = commit
        self.name = "TDBase"

        self._setup_lr_schedule()
        self.logger_kwargs = {} if logger_kwargs is None else logger_kwargs
        self.policy = self._create_policy(policy, policy_kwargs).to(self.device)

        self.H = horizon
        self.tau = tau
        self.gamma = gamma
        self.lamda = lamda

        self.policy_noise = policy_noise

        self._build()

    def _build(self):
        self._create_save_path()
        self.actor_batch_norm_stats = get_parameters_by_name(self.policy.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.policy.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.policy.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.policy.critic_target, ["running_"])
    def _create_save_path(self):
        index = 1
        path = f"{self.save_path}/{self.name}_{self.commit}_{index}" if self.commit is not None \
            else f"{self.save_path}/{self.name}_{index}"
        while os.path.exists(path):
            index += 1
            path = f"{self.save_path}/{self.name}_{self.commit}_{index}" if self.commit is not None \
                else f"{self.save_path}/{self.name}_{index}"
        self.policy_save_path = path

    def _create_logger(self,
                       format_strings=None,
                       ) -> logger.Logger:
        if format_strings is None:
            format_strings = ["stdout", "tensorboard"]
        l = logger.configure(self.policy_save_path, format_strings)
        return l

    def _create_policy(
            self,
            policy: Type[BasePolicy],
            policy_kwargs: Optional[Dict],
    ):
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        if isinstance(policy, str):
            policy_class = self.policy_aliases[policy]
        else:
            policy_class = policy

        policy = policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **policy_kwargs
        )


        return policy

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def log(self, timestep: int):
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
    ):
        assert self.H >= 1, "horizon must be greater than 1"
        self.policy.train()
        self._logger = self._create_logger(**self.logger_kwargs)

        buffer_len = 100
        # initialization
        eq_rewards_buffer, eq_len_buffer, eq_success_buffer = \
            deque(maxlen=buffer_len), deque(maxlen=buffer_len), deque(maxlen=buffer_len)

        for _ in range(buffer_len):
            eq_success_buffer.append(False)

        current_step, previous_step, previous_time = 0, 0, 0
        start_time = time.time()
        poses = []
        try:
            with tqdm(total=total_timesteps) as pbar:
                while current_step < total_timesteps:
                    actor_loss, critic_loss = 0., 0.  # th.tensor(0, device=self.device), th.tensor(0, device=self.device)
                    obs = self.env.get_observation()
                    # obs2["state"] = obs["state"].detach()

                    info_id_list = [i for i in range(self.num_envs)]
                    active_steps = th.zeros((self.num_envs,), device=self.device)
                    pre_active = th.ones((self.num_envs,), device=self.device, dtype=th.bool) # is this step effective or not
                    for inner_step in range(self.H):

                        # iteration
                        actions = self.policy.actor(obs)-0
                        actions += th.randn_like(actions, device=self.device) * self.policy_noise
                        clipped_actions = th.clip(
                            actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )

                        # calculate the value of the current step
                        with th.no_grad():
                            values_bp, _ = th.cat(self.policy.critic(obs, clipped_actions), dim=1).min(dim=1) #retain the gradient
                        values, _ = th.cat(self.policy.critic(obs.detach(), clipped_actions.detach()), dim=1).min(dim=1)

                        # step
                        obs, reward, done, info = self.env.step(clipped_actions)
                        reward = reward.to(self.device)

                        current_step += self.num_envs

                        # compute the temporal difference
                        discount_factor = self.lamda ** inner_step
                        next_actions = self.policy.actor_target(obs).clip(
                            th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )
                        next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=1).min(dim=1)
                        target = reward.detach() + self.gamma * next_values * discount_factor

                        # compute the loss
                        critic_loss += (values - target).pow(2) * discount_factor * pre_active
                        actor_loss += -1 * reward * discount_factor * pre_active
                        active_steps += pre_active
                        # r + next step value
                        if inner_step >= 1:
                            actor_loss += -1 * values_bp * self.gamma * discount_factor / self.lamda * pre_active

                        pre_active = (~done.to(self.device))

                        # record the reward and length of the episode
                        # as the finished process will not be reset, thus we only observe the status of the last step
                        for index in info_id_list:
                            if self.env.done[index]:
                                info_id_list.remove(index)
                                eq_rewards_buffer.append(self.env.info[index]["episode"]["r"])
                                eq_len_buffer.append(self.env.info[index]["episode"]["l"])
                                eq_success_buffer.append(self.env.info[index]["is_success"])

                    # the next value of the last step
                    actions = self.policy.actor(obs)-0
                    actions += th.randn_like(actions, device=self.device) * self.policy_noise
                    clipped_actions = th.clip(
                        actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                    )
                    with th.no_grad():
                        values_bp, _ = th.cat(self.policy.critic(obs, clipped_actions), dim=1).min(dim=1)
                    actor_loss += -1 * values_bp * discount_factor * pre_active

                    # update
                    critic_loss = (critic_loss / active_steps).mean()
                    actor_loss = (actor_loss / active_steps).mean()

                    self.policy.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    self.policy.critic.optimizer.step()

                    self.policy.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.policy.actor.optimizer.step()

                    polyak_update(params=self.policy.critic.parameters(), target_params=self.policy.critic_target.parameters(), tau=self.tau)
                    polyak_update(params=self.policy.actor.parameters(), target_params=self.policy.actor_target.parameters(), tau=self.tau)
                    polyak_update(params=self.critic_batch_norm_stats, target_params=self.critic_batch_norm_stats_target, tau=1.)
                    polyak_update(params=self.actor_batch_norm_stats, target_params=self.actor_batch_norm_stats_target, tau=1.)
                    # soft_update(self.policy.critic_target, self.policy.critic, self.tau)
                    # soft_update(self.policy.actor_target, self.policy.actor, self.tau)

                    if pbar.n - previous_step >= self._dump_step and len(eq_rewards_buffer) > 0:
                        self._logger.record("time/fps", (current_step - previous_step) / (time.time() - previous_time))
                        self._logger.record("rollout/ep_rew_mean", sum(eq_rewards_buffer) / len(eq_rewards_buffer))
                        self._logger.record("rollout/ep_len_mean", sum(eq_len_buffer) / len(eq_len_buffer))
                        self._logger.record("rollout/eq_success_rate", sum(eq_success_buffer) / len(eq_success_buffer))
                        self._logger.record("train/actor_loss", actor_loss.item())
                        self._logger.record("train/critic_loss", critic_loss.item())
                        self._logger.record("train/horizon_mean", active_steps.mean().item())
                        self._logger.dump(current_step)
                        previous_time, previous_step = time.time(), current_step
                    pbar.update((inner_step + 1) * self.num_envs)

                    # reset terminated agents
                    self.env.examine()
                    self.env.detach()

        except KeyboardInterrupt:
            pass

        return self.policy

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.policy_save_path
        th.save(self.policy, path + ".pth")
        print(f"Model saved at {path}")

    def predict(self, obs):
        self.policy.eval()
        obs = {key: th.as_tensor(value) for key, value in obs.items()}
        action = self.policy.get_action(obs)
        clipped_actions = th.clip(
            action, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
        )
        with th.no_grad():
            return clipped_actions

    # @staticmethod
    def load(self, path: Optional[str]):
        path += ".pth" if not path.endswith(".pth") else path
        self.policy = th.load(path).to(self.policy.device)
        return self

    @property
    def logger(self):
        return self._logger
