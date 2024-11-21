import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradientReversalFunction(Function):
    @staticmethod
    def forward(self, x, alpha):
        self.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        _, alpha = self.saved_tensors
        if self.needs_input_grad[0]:
            grad_input = -1 * alpha * grad_output
        return grad_input, None


revgrad = GradientReversalFunction.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class GRL(nn.Module):
    def __init__(self, balanced_rep_dim, num_policy, lambda_):
        super().__init__()

        self.grl = GradientReversal(lambda_)

        self.grl_l1 = nn.Linear(balanced_rep_dim, 128)
        self.grl_l2 = nn.Linear(128, 64)
        self.grl_l3 = nn.Linear(64, num_policy)

        self.grl_bn1 = nn.BatchNorm1d(128)
        self.grl_bn2 = nn.BatchNorm1d(64)

    def forward(self, balanced_representation):
        e = self.grl(balanced_representation)
        # e = balanced_representation
        e = F.elu(self.grl_bn1(self.grl_l1(e)))
        e = F.elu(self.grl_bn2(self.grl_l2(e)))
        e = F.softmax(self.grl_l3(e), dim=1)

        return e


class BalancingRepresentationLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.elu(self.fc(x))
        x = self.batch_norm(x)
        return x

    def enable_training(self, enable=True):
        for param in self.parameters():
            param.requires_grad = enable


class BalancingRepresentationCounterFactualNetwork(nn.Module):
    def __init__(self, num_policy, lambda_, br_input_dim, br_output_dim):
        super().__init__()

        self.br_layer = BalancingRepresentationLayer(br_input_dim, br_output_dim)
        self.grl_module = GRL(br_output_dim, num_policy, lambda_)

    def forward(self, pe_hidden_representation):
        br_h = self.br_layer(pe_hidden_representation)
        e = self.grl_module(br_h)

        return br_h, e


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class FAST_Q(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):
        self.br_network = BalancingRepresentationCounterFactualNetwork(
            num_policy=1,
            lambda_=0.1,
            br_input_dim=state_dim,
            br_output_dim=state_dim * 2,
        ).to(device)
        self.br_network_optimizer = torch.optim.Adam(
            self.br_network.parameters(),
            lr=3e-4,
        )

        self.actor = Actor(state_dim * 2, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim * 2, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        self.br_network.eval()
        with torch.no_grad():
            br_state, _ = self.br_network(state)
        self.br_network.train()
        return self.actor(br_state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Get BR Hidden State Representation and class prediction
        br_hidden_state, predicted_class = self.br_network(state)

        # Form the target class based on the policy
        target_class_0 = torch.full(
            size=(predicted_class.size(0),), fill_value=0.0, dtype=torch.float
        ).to(device)

        counterfactual_loss_0 = F.binary_cross_entropy(
            predicted_class.float(), target_class_0.unsqueeze(-1).float()
        )

        target_class_1 = torch.full(
            size=(predicted_class.size(0),), fill_value=1.0, dtype=torch.float
        ).to(device)

        counterfactual_loss_1 = F.binary_cross_entropy(
            predicted_class.float(), target_class_1.unsqueeze(-1).float()
        )
        counterfactual_loss = counterfactual_loss_0 + counterfactual_loss_1

        with torch.no_grad():
            br_state, _ = self.br_network(state)
            br_next_state, _ = self.br_network(next_state)

            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(br_next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(br_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(br_state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # l1 = counterfactual_loss.detach().item()
        # l2 = critic_loss.detach().item()
        # w1 = l2 / (l1 + l2)
        # w2 = l1 / (l1 + l2)
        # combined_loss = (w1 * counterfactual_loss) + (2 * w2 * critic_loss)

        combined_loss = counterfactual_loss + critic_loss

        # Optimize the critic
        self.br_network_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        combined_loss.backward()
        self.br_network_optimizer.step()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(br_state)
            Q = self.critic.Q1(br_state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
