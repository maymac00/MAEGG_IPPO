import copy

import numpy as np
import optuna
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging, Report2Optuna, \
    SaveCheckpoint
from IndependentPPO.lr_schedules import IndependentPPOAnnealing, DefaultPPOAnnealing
from IndependentPPO import IPPO, ParallelIPPO
from IndependentPPO.config import args_from_json
import gym
import matplotlib
from IndependentPPO.hypertuning import OptunaOptimizer, OptunaOptimizeMultiObjective


class OptimizerMAEGG(OptunaOptimizer):
    def __init__(self, direction, env_config, ppo_config, study_name=None, save=None, n_trials=1, pruner=None):
        super().__init__(direction, study_name, save, n_trials, pruner)
        self.env_config = env_config
        self.ppo_config = ppo_config

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
            Report2Optuna(ppo, trial, 1000),
            AnnealEntropy(ppo),
        ])
        return ppo

    def objective(self, trial):
        ppo = self.construct_ppo(trial)
        ppo.train()
        for k, v in ppo.__dict__.items():
            trial.set_user_attr(k, v)
        return self.eval_by_mean(ppo)

    def eval_by_mean(self, ppo, n=20):
        metric = 0
        ppo.eval_mode = True
        for i in range(n):  # Sim does n_steps so keep it low
            rec = ppo.rollout()
            metric += sum(rec) / rec.shape[0]
        metric /= n
        return metric

    def eval_by_loss(self, ppo):
        if len(ppo.run_metrics["mean_loss"]) < 10:
            return abs(np.array(ppo.run_metrics["mean_loss"][-1]).mean())
        return abs(np.array(ppo.run_metrics["mean_loss"][-10]).mean())

    def pre_objective_callback(self, trial):
        self.args = copy.deepcopy(self.ppo_config)
        self.args.save_dir += self.study_name
        self.args.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
        self.args.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
        self.args.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)

        # Set ppo parameters as user attributes.
        for k, v in self.args.__dict__.items():
            trial.set_user_attr(k, v)


class OptimizerMOMAGG(OptunaOptimizeMultiObjective):
    def __init__(self, direction, env_config, ppo_config, study_name=None, save=None, n_trials=1):
        super().__init__(direction, study_name, save, n_trials)
        self.env_config = env_config
        self.ppo_config = ppo_config

    def objective(self, trial):

        self.args = copy.deepcopy(self.ppo_config)
        self.args.save_dir += "/optuna/" + self.study_name
        self.args.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
        self.args.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
        self.args.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)

        # Set ppo parameters as user attributes.
        for k, v in self.args.__dict__.items():
            trial.set_user_attr(k, v)

        env = gym.make("MultiAgentEthicalGathering-v1", **self.env_config)
        env = NormalizeReward(env)

        # Set environment parameters as user attributes.
        for k, v in self.env_config.items():
            trial.set_user_attr(k, v)

        ppo = ParallelIPPO(self.args, env=env)
        ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=150),
            TensorBoardLogging(ppo, log_dir="jro/EGG_DATA"),
            AnnealEntropy(ppo, concavity=self.args.concavity_entropy),
        ])

        ppo.train()
        env.active = False
        return self.eval_mo(ppo)

    def eval_mo(self, ppo, n=20):
        metric = np.zeros(ppo.n_agents)
        ppo.eval_mode = True
        for i in range(n):  # Sim does n_steps so keep it low
            rec = ppo.rollout()
            metric += rec
        metric /= n
        return tuple(metric)


if __name__ == "__main__":
    args = args_from_json("hyperparameters/tiny.json")
    args.tot_steps = 15000
    opt1 = OptimizerMAEGG("maximize", tiny, args, n_trials=1, save=args.save_dir, study_name=args.tag)
    opt2 = OptimizerMOMAGG(["maximize", "maximize"], tiny, args, n_trials=1, save=args.save_dir,
                           study_name=args.tag + "_mo")

    opt1.optimize()
    opt2.optimize()
