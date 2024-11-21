import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class Config:
    device = "cuda"
    experiment = "exp_1"
    num_policies = 3

    # Running loops
    num_pretraining_epochs = 100
    num_epochs = 2000
    batch_size = 1024
    learning_rate = 3e-4
    l2_regularisation_coeff = 1e-5

    # Input data dimesions
    temporal_feature_dim = 78
    static_feature_dim = 10
    action_dim = 3
    br_input_dim = 128
    br_output_dim = 256

    # Policy Experts
    pe_hidden_dim = 64
    pe_static_hidden_dim = 32

    # Balancing representation layer Hyper-parameters
    br_input_dim = 128
    br_output_dim = 256
    lambda_ = 0.1

    # Critic
    decomposed_reward_dim = 3

    # =========================
    # Training Hyper-parameters
    # =========================

    # Q(s, a) = r + discount * Q(s', a')
    discount_factor = 0.7
    max_exploration_factor = 0.6

    # Soft update weightage of the target_actor/target_critic
    tau = 0.005

    policy_noise = 0.2
    noise_clip = 0.5

    # How frequently the actor network is updated
    policy_freq = 2

    # https://arxiv.org/pdf/2106.06860:
    # π(actor) = λ * Q(s, π(s)) − (π(s) − a) ** 2 where
    # λ = alpha/abs(Q(s, π(s))).mean()
    alpha = 2.5

    # Important Directories
    results_dir = RESULTS_DIR
    models_dir = MODELS_DIR
    data_folder = DATA_DIR

    # ===========================
    # Evaluation Hyper-parameters
    # ===========================
    eval_freq = 5
    eval_episodes = 10
    eval_epochs: list = [*range(300)]
