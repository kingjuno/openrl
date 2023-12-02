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
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch

from openrl.modules.model_config import ModelTrainConfig
from openrl.modules.networks.policy_network import PolicyNetwork
from openrl.modules.rl_module import RLModule
from openrl.modules.utils.util import update_linear_schedule


class HAPPOModule(RLModule):
    def __init__(
        self,
        cfg,
        input_space: gym.spaces.Box,
        act_space: gym.spaces.Box,
        device: Union[str, torch.device] = "cpu",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        model_dict: Optional[Dict[str, Any]] = None,
    ):
        # TODO: fix model_configs
        model_configs = {}
        model_configs["actor"] = ModelTrainConfig(
            lr=cfg.actor_lr,
            model=(
                model_dict["actor"]
                if model_dict and "actor" in model_dict
                else PolicyNetwork
            ),
            input_space=input_space,
        )
        super(HAPPOModule, self).__init__(
            cfg=cfg,
            model_configs=model_configs,
            act_space=act_space,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        self.obs_space = input_space

    def lr_decay(self, episode, episodes):
        # TODO: self.optimizer?
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        # TODO: self.models["actor"]? or in future self.model?
        actions, action_log_probs, rnn_states = self.models["actor"](
            obs, rnn_states, masks, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states

    def get_values(self):
        raise NotImplementedError

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        # TODO: self.models["actor"]? or in future self.model?
        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.models["actor"].evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        # TODO: self.models["actor"]? or in future self.model?
        actions, _, rnn_states = self.models["actor"](
            obs, rnn_states, masks, available_actions, deterministic
        )
        return actions, rnn_states

    @staticmethod
    def init_rnn_states(
        rollout_num: int, agent_num: int, rnn_layers: int, hidden_size: int
    ):
        masks = np.ones((rollout_num * agent_num, 1), dtype=np.float32)
        rnn_state = np.zeros((rollout_num * agent_num, rnn_layers, hidden_size))
        return rnn_state, masks
