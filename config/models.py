"""Model parameters for the experiment"""

# Model Parameters
DQN_PARAMS = {
    "update_frequency": 1,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.99995,
    "target_update": 2000,
    "memory_size": 50000,
    "batch_size": 128,
}

PPO_PARAMS = {
    "update_frequency": 1024,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "ppo_epochs": 5,
    "batch_size": 1024,
    "mini_batch_size": 128,
    "use_gae": True,
    "use_clipped_value_loss": True,
    "use_normalized_advantage": True,
    "use_trust_region": True,
    "use_adaptive_entropy_bonus": True,
}

A2C_PARAMS = {
    "update_frequency": 5,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
}

SAC_PARAMS = {
    "memory_size": 100000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 0.0003, 
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "automatic_entropy_tuning": True,
    "alpha": 0.2,
    "target_entropy": None,
    "update_frequency": 1,
}