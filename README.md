# FAST-Q: Fast-track Exploration with Adversarially Balanced State Representations for Counterfactual Action Estimation in Offline Reinforcement Learning

FAST-Q: Fast-track Exploration with Adversarially Balanced State Representations for Counterfactual Action Estimation in Offline Reinforcement Learning  _(under review at WWW 2025)_.

<img 
    src="img\fast_q_architecture.png"
    alt="FAST-Q Architecture" 
    style="height:400; margin-left:auto; margin-right:auto; display:block" 
/>
<div style="display:table; margin-left:auto; margin-right:auto"><p style="font-size:14pt; font-family:Verdana">FAST-Q Architecture</p></div>
  
<br>

## Abstract
Recent advancements in state-of-the-art (SOTA) offline reinforcement learning (RL) have primarily focused on addressing function approximation errors, which contribute to the overestimation of Q-values for out-of-distribution actions — a challenge that static datasets exacerbate. However, high-stakes applications such as recommendation systems in online gaming, introduce further complexities due to players’ psychology/intent driven by gameplay experiences and the platform’s inherent volatility. These factors create highly sparse, partially overlapping state spaces across policies, further influenced by the experiment path selection logic which biases state spaces towards specific policies. Current SOTA methods constrain learning from such offline data by clipping known counterfactual actions as out-of-distribution due to poor generalization across unobserved states. Further aggravating conservative Q-learning and necessitating more online exploration. 

FAST-Q introduces a novel approach that:  

1. Leverages Gradient Reversal Learning to construct balanced state representations, regularizing the policy-specific bias between the players’ state and action thereby enabling counterfactual estimation.  
2. Supports offline counterfactual exploration in parallel with static data exploitation.  
3. Proposes a Q-value decomposition strategy for multi-objective optimization, facilitating explainable recommendations over short and long-term objectives. 
  
These innovations demonstrate superiority of FAST-Q over prior SOTA approaches and demonstrates at least 0.15% increase in player returns, 2% improvement in lifetime value (LTV), 0.4% enhancement in the recommendation driven engagement, 2% improvement in the players’ platform dwell time and an impressive 10% reduction in the costs associated with the recommendation, on our volatile gaming platform.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

In order to install and run the model, you will need a working Python 3 distribution, and optionally a NVIDIA GPU with CUDA and cuDNN installed.

## Installing

In order to install the model and run it, you have to follow these steps:

* Clone the repository, i.e. run `git clone https://github.com/scarce-user-53/FAST-Q.git`
* Change into the directory, i.e. run `cd FAST-Q`
* Install the requirements:
  * run `chmod +x setup_env.sh`
  * run `sh setup_env.sh`
* Change into the code directory:
  * run `cd src` for training on our sample players data for multi-policy FAST-Q
  * run `cd single-policy-gym-tasks` for training on gym tasks for single policy FAST-Q

## Training
After successful installation, now you should be able to run the code on either the players' gameplay time-series data or the D4RL gym tasks.
1. For Players Data, the command is: `python train.py`. 
2. For D4RL gym tasks, the commands are:
   1. `chmod +x run_experiments-<environment_name>.sh`
   2. `sh run_experiments-<environment_name>.sh`


## Acknowledgement
We highly appreciate the following works for their valuable code and data for batch RL:

https://github.com/sfujim/TD3_BC

https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/
