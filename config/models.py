"""Model parameters for the experiment"""

# Model Parameters
DQN_PARAMS = {
    "update_frequency": 1,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9999,
    "target_update": 10,
    "memory_size": 10000,
    "batch_size": 128,
}

DDPG_PARAMS = {
    "update_frequency": 1,
    "learning_rate_actor": 0.0001,
    "learning_rate_critic": 0.001,
    "gamma": 0.99,
    "tau": 0.001,  # Target network update rate
    "memory_size": 10000,
    "batch_size": 128,
}

PPO_PARAMS = {
    "update_frequency": 1024,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "ppo_epochs": 10,
    "batch_size": 128,
    "mini_batch_size": 4,
    "use_gae": True,
    "use_clipped_value_loss": True,
    "use_normalized_advantage": True,
    "use_trust_region": True,
    "use_adaptive_entropy_bonus": True,
}

A2C_PARAMS = {
    "update_frequency": 5,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
}

 