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

from typing import Union

import numpy as np
import torch
import torch.nn as nn

from openrl.algorithms.base_algorithm import HABaseAlgorithm
from openrl.modules.utils.util import get_grad_norm
from openrl.utils.util import check


class HAPPOAlgorithm(HABaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_aggregation = cfg.action_aggregation
        super(HAPPOAlgorithm, self).__init__(cfg, init_module, agent_num, device)

    def happo_update(self, sample):
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        action_log_probs, dist_entropy, _ = self.algo_module.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss
        # TODO: check this afterwards
        # if its just a single model remove using dict?**
        self.algo_module.optimizer["actor"].zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term

        if self._use_max_grad_norm:
            # TODO: algo_module.models["actor"]?
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.algo_module.models["actor"].parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(
                self.algo_module.models["actor"].parameters()
            )

        self.algo_module.optimizer["actor"].step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, advantages, state_type):
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator_actor(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator_actor(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.feed_forward_generator_actor(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.happo_update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
