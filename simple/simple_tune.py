import ray.tune.registry as RLLibRegistry
from ray import tune
from ray.rllib.agents.a3c import DEFAULT_CONFIG
from ray.rllib.agents.a3c import a3c
from ray.rllib.models import ModelCatalog

from simple.simple_env import SimpleEnv
from simple.simple_model import SimpleModel

RLLibRegistry.register_env("SumoSimple-v0", lambda env_context: SimpleEnv.get_instance(render=False))
ModelCatalog.register_custom_model("simple_model", SimpleModel)

config = DEFAULT_CONFIG.copy()
config.update({
        "env": "SumoSimple-v0",
        "framework": "torch",
        "gamma": 0.99,

        "lr": tune.grid_search([1e-5, 1e-4, 1e-3]),
        "model": {
            "custom_model": "simple_model",
            "custom_model_config": {
                "hidden_size": tune.grid_search([32, 64, 128, 256, 512]),
            },
            "max_seq_len": 1
        },

        ## ROLLOUT WORKER ##
        "batch_mode": "complete_episodes",
        "num_workers": 31,
        "num_envs_per_worker": tune.grid_search([1, 3, 5]),
        "rollout_fragment_length": 100,

        ## OPTIMIZER ##
        "min_iter_time_s": 0.1,  # minimal iteration time in seconds
        "timesteps_per_iteration": 1000,
    })

tune.run(
    a3c.A3CTrainer,
    name="simple_experiment",
    stop={
        "timesteps_total": 100e6,  # Million steps
    },
    num_samples=1,
    config=config
)