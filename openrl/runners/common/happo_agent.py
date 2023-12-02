#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
from typing import Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch

from openrl.algorithms.base_algorithm import HABaseAlgorithm
from openrl.algorithms.happo import HAPPOAlgorithm
from openrl.buffers import OffPolicyReplayBuffer as ReplayBuffer
from openrl.buffers.utils.obs_data import ObsData
from openrl.drivers.base_driver import BaseDriver
from openrl.drivers.offpolicy_driver import OffPolicyDriver as Driver
from openrl.runners.common.base_agent import SelfAgent
from openrl.runners.common.rl_agent import RLAgent
from openrl.utils.logger import Logger
from openrl.utils.type_aliases import MaybeCallback
from openrl.utils.util import _t2n


class HAPPOAgent(RLAgent):
    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,
        env: Union[gym.Env, str] = None,
        run_dir: Optional[str] = None,
        env_num: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project_name: str = "HAPPOAgent",
    ) -> None:
        super(HAPPOAgent, self).__init__(
            net,
            env,
            run_dir,
            env_num,
            rank,
            world_size,
            use_wandb,
            use_tensorboard,
            project_name=project_name,
        )

    def train(
        self: SelfAgent,
        total_time_steps: int,
        callback: MaybeCallback = None,
        train_algo_class: Type[HABaseAlgorithm] = HAPPOAlgorithm,
        logger: Optional[Logger] = None,
        driver_class: Type[BaseDriver] = Driver,
    ) -> None:
        # TODO: implement buffers
        self.actor_buffer = None
        self.critic_buffer = None
        actor_train_infos = []
        # TODO: if episode_length is the real name
        factor = np.ones(
            (
                self._cfg.episode_length,
                self._cfg.n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )
        # TODO: do something with critic_buffer
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )
        if self._cfg.state_type == "FP":
            # TODO: num agents in cfg or self?
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        # TODO: whats fixed order?
        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())

        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, self.actor_buffer[agent_id].available_actions.shape[-1]),
            )

            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
