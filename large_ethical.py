import copy
import time
import gym
import numpy as np
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
        # Sleep for random time to avoid choosing the same arguments on case of concurrent runs.
        time.sleep(np.random.randint(0, 10))

    def pre_objective_callback(self, trial):
        self.args = copy.deepcopy(self.ppo_config)
        self.args.save_dir += "/" + self.study_name

    def construct_ppo(self, trial):
        time.sleep(np.random.randint(1, 10))
        env = gym.make("MultiAgentEthicalGathering-v1", **self.env_config)
        env = NormalizeReward(env)

        self.args.ent_coef = 0.12

        # We make groups of efficiency to reduce the amount of parameters to tune.
        eff_groups = [np.where(env.efficiency == value)[0].tolist() for value in np.unique(env.efficiency)]
        eff_dict = {}

        for k in eff_groups[0]:
            eff_dict[k] = {
                "actor_lr": trial.suggest_float(f"actor_lr_0", 1e-04, 6e-04, step=2e-04),
                "critic_lr": trial.suggest_float(f"critic_lr_0", 300e-04, 700e-04, step=200e-04)
            }
        for k in eff_groups[1]:
            eff_dict[k] = {
                "actor_lr": trial.suggest_float(f"actor_lr_1", 1e-04, 6e-04, step=2e-04),
                "critic_lr": trial.suggest_float(f"critic_lr_1", 300e-04, 700e-04, step=200e-04)
            }
        """
        for i, group in enumerate(eff_groups):
            actorlr = trial.suggest_float(f"actor_lr_{i}", 0.000001, 0.00001, step=0.000005)
            criticlr = trial.suggest_float(f"critic_lr_{i}", 0.0002, 0.001, step=0.0001)
            for agent in group:
                eff_dict[agent] = {"actor_lr": actorlr, "critic_lr": criticlr}
        """
        self.args.seed = trial.suggest_int("seed", 2, 20)

        self.args.concavity_entropy = 2.0
        final_value = 0.4

        # Do not modify args after this point.
        ppo = ParallelIPPO(self.args, env=env)
        ppo.lr_scheduler = IndependentPPOAnnealing(ppo, eff_dict)
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=5000, show_time=True),
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
    try:
        db = float(args.tag.split("db")[1].split("_")[0])
        we = float(args.tag.split("we")[1].split("_")[0])
        eff_rate = float(args.tag.split("effrate")[1].split("_")[0])
    except:
        raise ValueError("Please provide the we, effrate and db values in the tag.")

    large["efficiency"] = [0.85] * int(args.n_agents * eff_rate) + [0.2] * int(args.n_agents - eff_rate * args.n_agents)
    large["donation_capacity"] = db
    large["we"] = [1, we]

    print(large["efficiency"])

    try:
        opt = LargeSizeOptimize("maximize", large, args, n_trials=1, save=args.save_dir, study_name=args.tag)
        opt.optimize()
    except Exception as e:
        opt = LargeSizeOptimize("maximize", large, args, n_trials=1, save=args.save_dir, study_name=args.tag)
        opt.optimize()
