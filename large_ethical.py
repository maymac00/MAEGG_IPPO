import copy

import gym
from EthicalGatheringGame import NormalizeReward
from EthicalGatheringGame.presets import medium, large
from IndependentPPO import ParallelIPPO
from IndependentPPO.callbacks import TensorBoardLogging, SaveCheckpoint, Report2Optuna, AnnealEntropy, \
    PrintAverageReward
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

    def construct_ppo(self, trial):
        env = gym.make("MultiAgentEthicalGathering-v1", **self.env_config)
        env = NormalizeReward(env)

        # Set environment parameters as user attributes.
        for k, v in self.env_config.items():
            trial.set_user_attr(k, v)

        ppo = ParallelIPPO(self.args, env=env)
        self.args.ent_coef = trial.suggest_float("ent_coef", 0.04, 0.06, step=0.01)
        self.args.tot_steps = 40000000
        """ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
            k: {
                "actor_lr": trial.suggest_float(f"actor_lr_{k}", 0.00001, 0.0001, step=0.00001),
                "critic_lr": trial.suggest_float(f"critic_lr_{k}", 0.0002, 0.001, step=0.0001)
            } for k in range(self.args.n_agents)
        })"""
        ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
            0: {
                "actor_lr": 3e-05,
                "critic_lr": 0.0008
            },
            1: {
                "actor_lr": 3e-05,
                "critic_lr": 0.0008
            },
            2: {
                "actor_lr": 7e-05,
                "critic_lr": 0.0009
            },
            3: {
                "actor_lr": 3e-05,
                "critic_lr": 0.0008
            },
            4: {
                "actor_lr": 7e-05,
                "critic_lr": 0.0009
            }
        })

        self.args.concavity_entropy = trial.suggest_float("concavity_entropy", 0.0, 4.0, step=0.5)
        final_value = trial.suggest_float("final_value", 0.6, 1.0, step=0.2)
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=10000),
            TensorBoardLogging(ppo, log_dir=f"{args.save_dir}/{args.tag}/log/{ppo.run_name}", f=50),
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
    try:
        we = float(args.tag.split("we")[1].split("_")[0])
    except:
        we = 10
    large["we"] = [1, we]
    opt = LargeSizeOptimize("maximize", large, args, n_trials=1, save=args.save_dir, study_name=args.tag)
    opt.optimize()
