import numpy as np
import optuna
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging, Report2Optuna

from IndependentPPO.IPPO import IPPO
from IndependentPPO import IPPO, ParallelIPPO
from IndependentPPO.config import args_from_json
import gym
import matplotlib
from IndependentPPO.hypertuning import OptunaOptimizer


def objective(trial):
    args = args_from_json("hyperparameters/tiny.json")
    args.save_dir += "/optuna/" + args.tag
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    env = NormalizeReward(env)

    for k, v in args.items():
        trial.set_user_attr(k, v)

    args.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
    args.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
    args.tot_steps = trial.suggest_int("tot_steps", 15000000, 25000000, step=5000000)
    args.h_layers = trial.suggest_int("h_layers", 2, 3)
    args.h_size = trial.suggest_int("h_size", 128, 256, step=128)
    ppo = ParallelIPPO(args, env=env)
    ppo.addCallbacks([
        PrintAverageReward(ppo, n=500),
        TensorBoardLogging(ppo, log_dir="jro/EGG_DATA/optuna"),
        AnnealEntropy(ppo, concavity=args.concavity_entropy),
    ])
    trial.set_user_attr("run_name", ppo.run_name)
    ppo.train()
    trial.set_user_attr("save_dir", ppo.folder)
    metric = 0
    ppo.eval_mode = True
    for i in range(20):  # Sim does n_steps so keep it low
        rec = ppo.rollout()
        metric += rec.mean()
    metric /= 20
    return metric


class OptimizerMAEGG(OptunaOptimizer):
    def __init__(self, direction, study_name=None, save=None, n_trials=1, pruner=None):
        super().__init__(direction, study_name, save, n_trials, pruner)

    def objective(self, trial):

        env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
        env = NormalizeReward(env)

        # Set environment parameters as user attributes.
        for k, v in tiny.items():
            trial.set_user_attr(k, v)

        ppo = ParallelIPPO(self.args, env=env)
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=150),
            # TensorBoardLogging(ppo, log_dir="jro/EGG_DATA"),
            Report2Optuna(ppo, trial, 1),
            AnnealEntropy(ppo),
        ])

        ppo.train()
        return self.eval_by_mean(ppo)

    def eval_by_mean(self, ppo, n=20):
        metric = 0
        ppo.eval_mode = True
        for i in range(n):  # Sim does n_steps so keep it low
            rec = ppo.rollout()
            metric += sum(rec) / rec.shape[0]
        metric /= n
        return metric

    def eval_by_product(self, ppo, n=20, offset=1000):
        # We should be careful if the reward is negative
        metric = 1
        ppo.eval_mode = True
        for i in range(n):  # Sim does n_steps so keep it low
            rec = ppo.rollout()
            metric += (rec+(np.ones_like(rec)*1000)).prod()
        metric /= n
        return metric

    def pre_objective_callback(self, trial):
        self.args = args_from_json("hyperparameters/tiny.json")
        self.args.save_dir += "/optuna/" + self.args.tag
        self.args.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
        self.args.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
        self.args.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)
        self.args.concavity_entropy = trial.suggest_float("concavity-entropy", 1.0, 3.5)

        # Set environment parameters as user attributes.
        for k, v in tiny.items():
            self.args = args_from_json("hyperparameters/tiny.json")
            self.args.save_dir += "/optuna/" + self.args.tag
            trial.set_user_attr(k, v)

if __name__ == "__class__":
    opt = OptimizerMAEGG("maximize", n_trials=1, save="