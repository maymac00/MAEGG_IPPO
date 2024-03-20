import copy

import gym
from EthicalGatheringGame import NormalizeReward
from EthicalGatheringGame.presets import large

from IndependentPPO.callbacks import TensorBoardLogging, SaveCheckpoint, Report2Optuna, AnnealEntropy, \
    PrintAverageReward
from IndependentPPO import ParallelCIPPO
from IndependentPPO.lr_schedules import DefaultPPOAnnealing, IndependentPPOAnnealing

from hyper_tuning import OptimizerMAEGG, OptimizerMOMAGG
from IndependentPPO.config import args_from_json


class LargeSizeOptimize(OptimizerMAEGG):
    def __init__(self, direction, env_config, ppo_config, study_name=None, save=None, n_trials=1, pruner=None):
        super().__init__(direction, env_config, ppo_config, study_name, save, n_trials,
                         pruner)

    def pre_objective_callback(self, trial):
        self.args = copy.deepcopy(self.ppo_config)
        self.args.save_dir += "/" + self.study_name
        self.args.const_limit_1 = 3
        self.args.const_limit_2= 3
        # self.args.ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, step=0.001)
        self.args.ent_coef = 0.02
        self.args.tot_steps = 30000000

    def construct_ppo(self, trial):
        env = gym.make("MultiAgentEthicalGathering-v1", **self.env_config)
        env = NormalizeReward(env)

        # Set environment parameters as user attributes.
        for k, v in self.env_config.items():
            trial.set_user_attr(k, v)

        ppo = ParallelCIPPO(self.args, env=env)

        ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
            k: {
                "actor_lr": trial.suggest_float(f"actor_lr_{k}", 0.00001, 0.0001, step=0.00001),
                "critic_lr": trial.suggest_float(f"critic_lr_{k}", 0.0002, 0.001, step=0.0001)
            } for k in range(self.args.n_agents)
        })

        # self.args.mult_lr = trial.suggest_float("lagran_multi_lr", 0.005, 0.07, step=0.015)
        # self.args.mult_init = trial.suggest_float("lagran_multi_init", 0.3, 0.8, step=0.1)
        self.args.concavity_entropy = 2.0
        final_value = trial.suggest_float("final_value", 0.0, 1.0, step=0.2)
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=5000),
            TensorBoardLogging(ppo, log_dir=f"{args.save_dir}/{args.tag}/log/{ppo.run_name}", f=50),
            SaveCheckpoint(ppo, 5000),
            AnnealEntropy(ppo, final_value=final_value, concavity=self.args.concavity_entropy),
        ])
        return ppo

    def objective(self, trial):
        ppo = self.construct_ppo(trial)
        print(trial.params)
        ppo.train()
        for k, v in vars(ppo.init_args).items():
            trial.set_user_attr(k, v)
        return self.eval_by_mean(ppo)


if __name__ == "__main__":
    args = args_from_json("hyperparameters/large.json")
    # parse we from the args.tag string. Example: "mediumwe0.9_try1" -> we = 0.9
    large["efficiency"] = [0.2] * 3 + [0.85] * 2
    large["donation_capacity"] = 10
    opt = LargeSizeOptimize("maximize", large, args, n_trials=1, save=args.save_dir, study_name=args.tag)
    opt.optimize()
