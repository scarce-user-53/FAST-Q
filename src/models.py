import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

sys.path.append("..")
from configuration import Config

config = Config()


class SampleDrop(nn.Module):
    """Applies dropout to input samples with a fixed mask."""

    def __init__(self, dropout=0):
        super().__init__()

        assert 0 <= dropout < 1
        self._mask = None
        self._dropout = dropout

    def set_weights(self, X):
        """Calculates a new dropout mask."""
        assert len(X.shape) == 2

        mask = Variable(torch.ones(X.size(0), X.size(1)), requires_grad=False)

        if X.is_cuda:
            mask = mask.cuda()

        self._mask = F.dropout(mask, p=self._dropout, training=self.training)

    def forward(self, X):
        """Applies dropout to the input X."""
        if not self.training or not self._dropout:
            return X
        else:
            return X * self._mask


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        n_layers,
        hidden_size,
        dropout_i=0,
        dropout_h=0,
        return_states=False,
    ):
        super(LSTMModel, self).__init__()

        assert all([0 <= x < 1 for x in [dropout_i, dropout_h]])
        assert all([0 < x for x in [input_size, n_layers, hidden_size]])
        assert isinstance(return_states, bool)

        self._input_size = input_size
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._dropout_i = dropout_i
        self._dropout_h = dropout_h
        self._return_states = return_states

        cells = []
        for i in range(n_layers):
            cells.append(nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=True))

        self._cells = nn.ModuleList(cells)
        self._input_drop = SampleDrop(dropout=self._dropout_i)
        self._state_drop = SampleDrop(dropout=self._dropout_h)

    def _new_state(self, batch_size):
        """Initalizes states."""
        h = Variable(torch.zeros(batch_size, self._hidden_size))
        c = Variable(torch.zeros(batch_size, self._hidden_size))

        return (h, c)

    def forward(self, X):
        """Forward pass through the LSTM."""
        X = X.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, input_size]
        seq_len, batch_size, input_size = X.shape

        all_ht = []
        all_h = []
        all_c = []

        for cell in self._cells:
            ht, ct = [], []

            # Initialize new state.
            h, c = self._new_state(batch_size)
            h = h.to(X.device)
            c = c.to(X.device)

            # Fix dropout weights for this cell.
            self._input_drop.set_weights(X[0, ...])  # Removes time dimension.
            self._state_drop.set_weights(h)

            for sample in X:
                h, c = cell(self._input_drop(sample), (self._state_drop(h), c))
                ht.append(h)
                ct.append(c)

            # Append final hidden state and cell state for this layer
            all_h.append(h)
            all_c.append(c)

            # Collect hidden states for the entire sequence [seq_len, batch_size, hidden_size]
            ht = torch.stack(ht, dim=0).permute(1, 0, 2)
            X = ht.clone().permute(1, 0, 2)  # Input for the next layer.
            all_ht.append(ht)

        all_h = torch.stack(all_h, dim=0)  # Shape: [n_layers, batch_size, hidden_size]
        all_c = torch.stack(all_c, dim=0)  # Shape: [n_layers, batch_size, hidden_size]

        return ht, (all_h, all_c)


class VarDropoutBasedLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        static_feature_input_dim,
        static_feature_output_dim,
        device,
        dropout_i=0.1,
        dropout_h=0.1
    ):
        super().__init__()
        self.device = device
        self.lstm = LSTMModel(
            input_size,
            n_layers=1,
            hidden_size=hidden_size,
            dropout_i=dropout_i,
            dropout_h=dropout_h,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.dense_static = nn.Linear(
            static_feature_input_dim, static_feature_output_dim
        )
        self.bn2 = nn.BatchNorm1d(static_feature_output_dim)

        # Defining the three MLPs, each with 3 layers and an output dimension of 3
        self.pe1_layer1 = nn.Linear(hidden_size + static_feature_output_dim, 256)
        self.pe1_layer2 = nn.Linear(256, 128)
        self.pe1_bn = nn.BatchNorm1d(128) 
        self.pe1_output_layer = nn.Linear(128, 3)

        self.pe2_layer1 = nn.Linear(hidden_size + static_feature_output_dim, 256)
        self.pe2_layer2 = nn.Linear(256, 128)
        self.pe2_bn = nn.BatchNorm1d(128)
        self.pe2_output_layer = nn.Linear(128, 3)

        self.pe3_layer1 = nn.Linear(hidden_size + static_feature_output_dim, 256)
        self.pe3_layer2 = nn.Linear(256, 128)
        self.pe3_bn = nn.BatchNorm1d(128)
        self.pe3_output_layer = nn.Linear(128, 3)

    def forward(self, x, x_static):
        x = x.to(self.device)

        x_static = x_static.to(self.device)
        x_static = torch.relu(self.dense_static(x_static))
        x_static = self.bn2(x_static)

        x, (h, c) = self.lstm(x)
        h = self.bn1(h[-1])

        # Concatenating hidden representation with static feature to get the state representation
        state = torch.cat((h, x_static), axis=1)

        # Policy Expert 1
        pe1_hidden1 = torch.relu(self.pe1_layer1(state))
        pe1_hidden2 = torch.relu(self.pe1_layer2(pe1_hidden1))
        pe1_hidden2 = self.pe1_bn(pe1_hidden2)  
        pe1_output = self.pe1_output_layer(pe1_hidden2)

        # Policy Expert 2
        pe2_hidden1 = torch.relu(self.pe2_layer1(state))
        pe2_hidden2 = torch.relu(self.pe2_layer2(pe2_hidden1))
        pe2_hidden2 = self.pe2_bn(pe2_hidden2)
        pe2_output = self.pe2_output_layer(pe2_hidden2)

        # Policy Expert 3
        pe3_hidden1 = torch.relu(self.pe3_layer1(state))
        pe3_hidden2 = torch.relu(self.pe3_layer2(pe3_hidden1))
        pe3_hidden2 = self.pe3_bn(pe3_hidden2)
        pe3_output = self.pe3_output_layer(pe3_hidden2)

        return pe1_hidden2, pe1_output, pe2_hidden2, pe2_output, pe3_hidden2, pe3_output


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

        self.grl_l1 = nn.Linear(balanced_rep_dim, 512)
        self.grl_l2 = nn.Linear(512, 256)
        self.grl_l3 = nn.Linear(256, num_policy)

        self.grl_bn1 = nn.BatchNorm1d(512)
        self.grl_bn2 = nn.BatchNorm1d(256)

    def forward(self, balanced_representation):
        e = self.grl(balanced_representation)
        e = F.elu(self.grl_bn1(self.grl_l1(e)))
        e = F.elu(self.grl_bn2(self.grl_l2(e)))
        # e = F.softmax(self.grl_l3(e), dim=1)
        e = F.sigmoid(self.grl_l3(e))

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
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        action = torch.relu(self.l1(state))
        action = torch.relu(self.l2(action))
        action = self.l3(action)

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, decomposed_reward_dim):
        super(Critic, self).__init__()

        """
        Following is the implementation from the https://arxiv.org/pdf/2106.06860 paper which uses two Q-network and behaviour cloning to train the Q-network. 
        params:

            state_dim: input dimension of the state of the player
            action_dim: output dimension of the action value
        
        returns:
            q-values corresponding to each q-network

        """

        # Q1 architecture
        # instead of state_dim call it BR_counterfactual_dim
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)
        self.l7 = nn.Linear(256, decomposed_reward_dim + 1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, 1)
        self.l9 = nn.Linear(256, decomposed_reward_dim + 1)

        # Batch Normalization
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.bn1(self.l1(sa)))
        q1_pen = F.relu(self.bn2(self.l2(q1)))
        q1 = self.l3(q1_pen)
        w1 = F.softmax(self.l7(q1_pen), dim=1)

        q2 = F.relu(self.bn3(self.l4(sa)))
        q2_pen = F.relu(self.bn4(self.l5(q2)))
        q2 = self.l6(q2_pen)
        w2 = F.softmax(self.l9(q2_pen), dim=1)
        # w1 and w2 are softmax of size (batch_size, reward_dim + 1)
        return q1, w1, q2, w2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.bn1(self.l1(sa)))
        q1_pen = F.relu(self.bn2(self.l2(q1)))
        q1 = self.l3(q1_pen)
        w1 = F.softmax(self.l7(q1_pen), dim=1)

        return q1, w1


class CounterfactualActorNetwork(nn.Module):
    def __init__(self, br_output_dim, action_dim):
        super().__init__()
        self.actor = Actor(br_output_dim, action_dim)

    def forward(self, br_counterfactual_hidden_state):
        action = self.actor(br_counterfactual_hidden_state)
        return action


class CounterfactualCriticNetwork(nn.Module):
    def __init__(self, br_output_dim, action_dim, decomposed_reward_dim):
        super().__init__()

        # Critic Network
        self.critic = Critic(br_output_dim, action_dim, decomposed_reward_dim)

    def forward(self, br_counterfactual_hidden_state, action):

        # Passing balancing representation(state) and action through the critic.
        q1, w1, q2, w2 = self.critic(br_counterfactual_hidden_state, action)

        return q1, w1, q2, w2

    def Q1(self, state, action):
        return self.critic.Q1(state, action)
