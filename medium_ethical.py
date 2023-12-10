import copy

import gym
from EthicalGatheringGame import NormalizeReward
from EthicalGatheringGame.presets import medium
from IndependentPPO import ParallelIPPO
from IndependentPPO.callbacks import TensorBoardLogging, SaveCheckpoint, Report2Optuna, AnnealEntropy
from IndependentPPO.lr_schedules import DefaultPPOAnnealing

from hyper_tuning import OptimizerMAEGG, OptimizerMOMAGG
from IndependentPPO.config import args_from_json


class MediumSizeOptimize(OptimizerMAEGG):
    def __init__(self, direction, env_config, ppo_config, study_name=None, save=None, n_trials=1, pruner=None):
        super().__init__(direction, env_config, ppo_config, study_name, save, n_trials,
                         pruner)

    def pre_objective_callback(self, trial):
        self.args = copy.deepcopy(self.ppo_config)
        self.args.save_dir += "/" + self.study_name
        self.args.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001, step=0.000005)
        self.args.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01, step=0.00005)
        self.args.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, step=0.0005)
        self.args.tot_steps = trial.suggest_int("tot_steps", 20000000, 100000000, step=20000000)
        self.args.h_layers = trial.suggest_int("h_layers", 2, 5, step=1)
        self.args.h_size = trial.suggest_int("h_size", 64, 256, step=64)
        self.args.concavity_entropy = trial.suggest_float("concavity_entropy", 1.0, 3.5, step=0.5)

    def construct_ppo(self, trial):
        env = gym.make("MultiAgentEthicalGathering-v1", **self.env_config)
        env = NormalizeReward(env)

        # Set environment parameters as user attributes.
        for k, v in self.env_config.items():
            trial.set_user_attr(k, v)

        ppo = ParallelIPPO(self.args, env=env)
        ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
        ppo.addCallbacks([
            # PrintAverageReward(ppo, n=150),
            TensorBoardLogging(ppo, log_dir="jro/EGG_DATA"),
            SaveCheckpoint(ppo, 1000),
            Report2Optuna(ppo, trial, 1000, type="mean_loss"),
            AnnealEntropy(ppo),
        ])
        return ppo

    def objective(self, trial):
        ppo = self.construct_ppo(trial)
        ppo.train()
        for k, v in vars(ppo.init_args).items():
            trial.set_user_attr(k, v)
        return self.eval_by_loss(ppo)


if __name__ == "__main__":
    args = args_from_json("hyperparameters/medium.json")
    medium["we"] = [1, 99]
    opt = MediumSizeOptimize("minimize", medium, args, n_trials=1, save=args.save_dir, study_name=args.tag)
    opt.optimize()
