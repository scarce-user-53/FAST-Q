import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

from configuration import Config

config = Config()


def save_model_obj(model: object, save_location: str) -> None:
    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    with open(save_location, "wb") as f:
        pickle.dump(model.state_dict(), f)


def load_model_obj(load_location: str) -> object:
    with open(load_location, "rb") as f:
        model_obj = pickle.load(f)

    return model_obj


def get_data(data_folder="data"):
    """
    1. state_temporal: [30, 78] dimensional feature represenation of the player at t timestep
    2. state_static: 10 dimensional static features of the player
    3. action: 3 * num_policies dimesional
    4. next_state_temporal: [30, 78] dimensional feature represenation of the player at t+1 timestep
    5. next_state_static: 10 dimensional static features of the player at t+1 timestep
    6. reward
    7. policy
    """
    (
        state_temporal,
        state_static,
        actions,
        next_state_temporal,
        next_state_static,
        reward_components,
        policy,
    ) = np.load(os.path.join(data_folder, "sample_data.npy"), allow_pickle=True)

    dataset = TensorDataset(
        torch.tensor(state_temporal, dtype=torch.float32),
        torch.tensor(state_static, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(next_state_temporal, dtype=torch.float32),
        torch.tensor(next_state_static, dtype=torch.float32),
        torch.tensor(reward_components, dtype=torch.float32),
        torch.tensor(policy, dtype=torch.float32),
    )
    return dataset


def count_trainable_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_title(text, decorator="="):
    num_decorators = len(text)

    print(decorator * num_decorators)
    print(text)
    print(decorator * num_decorators)


def print_model_info(model, model_name):
    model_name = model_name.replace("_", " ").strip().title()
    print_title(model_name)

    print(
        "\nTrainable model parameters: {0:.4f} million\n".format(
            count_trainable_parameters(model) / 1e6
        )
    )

    print("Model Architecture:")
    print(model, end="\n\n")
