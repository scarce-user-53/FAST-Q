import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from configuration import Config
from helpers import get_data, load_model_obj, print_model_info, save_model_obj
from models import (
    BalancingRepresentationCounterFactualNetwork,
    CounterfactualActorNetwork,
    CounterfactualCriticNetwork,
    VarDropoutBasedLSTM,
)

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_cores = os.cpu_count()
torch.set_num_threads(cpu_cores)


def get_policy_experts(load_weights: bool = False):
    policy_experts = VarDropoutBasedLSTM(
        input_size=config.temporal_feature_dim,
        hidden_size=config.pe_hidden_dim,
        static_feature_input_dim=config.static_feature_dim,
        static_feature_output_dim=config.pe_static_hidden_dim,
        device=device,
    ).to(device)

    policy_experts_optimizer = torch.optim.Adam(
        policy_experts.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_regularisation_coeff,
    )

    print_model_info(policy_experts, "Policy Experts")

    if load_weights:
        weights = load_model_obj(os.path.join(config.models_dir, "policy_experts.pt"))
        policy_experts.load_state_dict(weights)
        return policy_experts

    return policy_experts, policy_experts_optimizer


def get_br_cf_network(load_weights: bool = False):
    br_counterfactual_network = BalancingRepresentationCounterFactualNetwork(
        num_policy=3,
        lambda_=config.lambda_,
        br_input_dim=config.br_input_dim,
        br_output_dim=config.br_output_dim,
    ).to(device)

    br_counterfactual_network_optimiser = torch.optim.Adam(
        br_counterfactual_network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_regularisation_coeff,
    )
    print_model_info(br_counterfactual_network, "Balancing Representation Counterfactual Network")

    if load_weights:
        weights = load_model_obj(os.path.join(config.models_dir, "br_cf_network.pt"))
        br_counterfactual_network.load_state_dict(weights)
        return br_counterfactual_network

    return br_counterfactual_network, br_counterfactual_network_optimiser


def train_policy_experts_and_br_network():
    policy_experts, policy_experts_optimizer = get_policy_experts()
    br_counterfactual_network, br_counterfactual_network_optimiser = get_br_cf_network()

    data_bincount_distribution = torch.ones(3, dtype=torch.float)

    print("Training Policy Experts and Counterfactual Balancing Representation Networks")
    for epoch in range(config.num_pretraining_epochs):
        print(f"Epoch: {epoch}")
        policy_experts.train()
        br_counterfactual_network.train()

        train_dataset = get_data(config.data_folder)

        class_weights = data_bincount_distribution.float()
        policy = torch.stack([item[-1] for item in train_dataset]).long()
        sample_weights = class_weights[policy]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=cpu_cores,
        )

        overall_bincount = torch.zeros(3, dtype=torch.long).to(device)

        for (
            state_temporal,
            state_static,
            actions,
            _,
            _,
            _,
            current_policy,
        ) in train_dataloader:
            state_temporal = state_temporal.to(device)
            state_static = state_static.to(device)
            actions = actions.to(device)
            current_policy = current_policy.to(device)

            (pe_h1, a_est1, pe_h2, a_est2, pe_h3, a_est3) = policy_experts(
                state_temporal, state_static
            )

            pe1_loss = F.mse_loss(a_est1, actions[:, 0:3])
            pe2_loss = F.mse_loss(a_est2, actions[:, 3:6])
            pe3_loss = F.mse_loss(a_est3, actions[:, 6:9])

            policy_experts_loss = pe1_loss + pe2_loss + pe3_loss

            _, e1 = br_counterfactual_network(pe_h1)
            _, e2 = br_counterfactual_network(pe_h2)
            _, e3 = br_counterfactual_network(pe_h3)

            e1_pred = torch.argmax(e1, dim=1)
            e2_pred = torch.argmax(e2, dim=1)
            e3_pred = torch.argmax(e3, dim=1)

            all_preds = torch.cat([e1_pred, e2_pred, e3_pred])
            overall_bincount += torch.bincount(all_preds, minlength=3)

            target_pe1 = torch.full((e3.size(0),), 0, dtype=torch.long).to(device)
            target_pe2 = torch.full((e3.size(0),), 1, dtype=torch.long).to(device)
            target_pe3 = torch.full((e3.size(0),), 2, dtype=torch.long).to(device)

            cross_entropy_loss_pe1 = F.cross_entropy(e1, target_pe1)
            cross_entropy_loss_pe2 = F.cross_entropy(e2, target_pe2)
            cross_entropy_loss_pe3 = F.cross_entropy(e3, target_pe3)

            counterfactual_loss = (
                cross_entropy_loss_pe1 + cross_entropy_loss_pe2 + cross_entropy_loss_pe3
            )

            combined_loss = policy_experts_loss + counterfactual_loss

            policy_experts_optimizer.zero_grad()
            br_counterfactual_network_optimiser.zero_grad()
            combined_loss.backward()
            policy_experts_optimizer.step()
            br_counterfactual_network_optimiser.step()

        data_bincount_distribution = overall_bincount.float() / overall_bincount.sum() * 100

    save_model_obj(policy_experts, os.path.join(config.models_dir, "policy_experts.pt"))
    save_model_obj(br_counterfactual_network, os.path.join(config.models_dir, "br_cf_network.pt"))


def train_actor_critic():
    policy_experts = get_policy_experts(load_weights=True)
    br_cf_network = get_br_cf_network(load_weights=True)

    critic = CounterfactualCriticNetwork(
        br_output_dim=config.br_output_dim,
        action_dim=config.action_dim,
        decomposed_reward_dim=config.decomposed_reward_dim,
    ).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.learning_rate, weight_decay=config.l2_regularisation_coeff
    )
    print_model_info(critic, "Critic Network")

    actor = CounterfactualActorNetwork(
        br_output_dim=config.br_output_dim, action_dim=config.action_dim
    ).to(device)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, weight_decay=config.l2_regularisation_coeff
    )
    print_model_info(actor, "Actor Network")

    print("Training Actor and Critic Networks")
    for epoch in range(config.num_epochs):
        print(f"Epoch: {epoch}")
        policy_experts.eval()
        br_cf_network.eval()
        critic.train()
        actor.train()

        # Initializing a data generator which will load each numpy chunk from s3 and yield it to the model
        train_dataset = get_data(config.data_folder)
        data_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=cpu_cores,
            shuffle=True,
        )

        for (
            state_temporal,
            state_static,
            actions,
            next_state_temporal,
            next_state_static,
            reward_components,
            current_policy,
        ) in data_loader:
            state_temporal = state_temporal.to(device)
            state_static = state_static.to(device)
            actions = actions.to(device)
            next_state_temporal = next_state_temporal.to(device)
            next_state_static = next_state_static.to(device)
            reward_components = reward_components.to(device)
            current_policy = current_policy.to(device)

            with torch.no_grad():
                (pe_h1, a_est1, pe_h2, a_est2, pe_h3, a_est3) = policy_experts(
                    state_temporal, state_static
                )

                br1_h, _ = br_cf_network(pe_h1)
                br2_h, _ = br_cf_network(pe_h2)
                br3_h, _ = br_cf_network(pe_h3)

                current_state_brs = torch.stack([br1_h, br2_h, br3_h], dim=0)
                state_br_h = current_state_brs[
                    current_policy.view(-1).long(), torch.arange(current_policy.size(0))
                ]

                (pe_h1_next, _, pe_h2_next, _, pe_h3_next, _) = policy_experts(
                    next_state_temporal, next_state_static
                )

                next_br1_h, _ = br_cf_network(pe_h1_next)
                next_br2_h, _ = br_cf_network(pe_h2_next)
                next_br3_h, _ = br_cf_network(pe_h3_next)

                next_state_brs = torch.stack([next_br1_h, next_br2_h, next_br3_h], dim=0)
                next_state_br_h = next_state_brs[
                    current_policy.view(-1).long(), torch.arange(current_policy.size(0))
                ]

            current_a = np.zeros((current_policy.size(0), 3))
            for i in range(current_policy.size(0)):
                policy_value = int(current_policy[i].item())
                start_index = policy_value * 3
                current_a[i, :] = actions[i].cpu().numpy()[start_index : start_index + 3]
            current_a = torch.from_numpy(current_a).float().to(device)

            with torch.no_grad():
                noise = (
                    torch.randn_like(current_a[:, :3], device=device) * config.policy_noise
                ).clamp(-config.noise_clip, config.noise_clip)

                next_action = actor_target(next_state_br_h) + noise
                target_Q1, _, target_Q2, _ = critic_target(next_state_br_h, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = (
                    0.25
                    * (
                        reward_components[:, 0].unsqueeze(1)
                        + reward_components[:, 1].unsqueeze(1)
                        - reward_components[:, 2].unsqueeze(1)
                        + reward_components[:, 3].unsqueeze(1)
                    )
                    + config.discount_factor * target_Q
                )

            current_Q1, w1, current_Q2, w2 = critic(state_br_h, current_a)
            q_value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            r1_loss = F.mse_loss(
                w1[:, 0].unsqueeze(1) * current_Q1, reward_components[:, 0].unsqueeze(1)
            ) + F.mse_loss(w2[:, 0].unsqueeze(1) * current_Q2, reward_components[:, 0].unsqueeze(1))
            r2_loss = F.mse_loss(
                w1[:, 1].unsqueeze(1) * current_Q1, reward_components[:, 1].unsqueeze(1)
            ) + F.mse_loss(w2[:, 1].unsqueeze(1) * current_Q2, reward_components[:, 1].unsqueeze(1))
            r3_loss = F.mse_loss(
                w1[:, 2].unsqueeze(1) * current_Q1, reward_components[:, 3].unsqueeze(1)
            ) + F.mse_loss(w2[:, 2].unsqueeze(1) * current_Q2, reward_components[:, 3].unsqueeze(1))

            critic_loss = 0.75 * q_value_loss + 0.25 * (r1_loss + r2_loss + r3_loss) / 3

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if (epoch + 1) % config.policy_freq == 0:
                exploration_factor = config.max_exploration_factor * min(
                    1.0, epoch / config.num_epochs
                )
                exploit = random.random() > exploration_factor

                if exploit:
                    pi = actor(state_br_h)
                    Q, _ = critic.Q1(state_br_h, pi)
                    lmbda = config.alpha / Q.abs().mean().detach()

                    actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, current_a)
                else:
                    random_policy = torch.randint(0, 3, (current_policy.size(0),))
                    state_br_h = current_state_brs[
                        random_policy, torch.arange(random_policy.size(0))
                    ]
                    target_a = np.zeros((random_policy.size(0), 3))
                    for i in range(random_policy.size(0)):
                        policy_value = int(random_policy[i].item())
                        target_a[i, :] = [
                            a_est1.detach().cpu().numpy()[i, :], 
                            a_est2.detach().cpu().numpy()[i, :], 
                            a_est3.detach().cpu().numpy()[i, :]
                        ][policy_value]
                    target_a = torch.from_numpy(target_a).float().to(device)
                    pi = actor(state_br_h)
                    Q, _ = critic.Q1(state_br_h, pi)
                    lmbda = config.alpha / Q.abs().mean().detach()

                    actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, target_a)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )

    save_model_obj(critic, os.path.join(config.models_dir, "critic.pt"))
    save_model_obj(actor, os.path.join(config.models_dir, "actor.pt"))


def main():
    train_policy_experts_and_br_network()
    train_actor_critic()


if __name__ == "__main__":
    main()
