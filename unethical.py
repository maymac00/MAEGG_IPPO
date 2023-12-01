import copy

import optuna
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging
from IndependentPPO import IPPO, ParallelIPPO
from IndependentPPO.config import args_from_json
import gym
import matplotlib
from IndependentPPO.lr_schedules import DefaultPPOAnnealing, IndependentPPOAnnealing
from hyper_tuning import OptimizerMAEGG, OptimizerMOMAGG


class Unethical(OptimizerMOMAGG):
    def __init__(self, direction, env_config, ppo_config, study_name=None, save=None, n_trials=1):
        super().__init__(direction, env_config, ppo_config, study_name, save, n_trials)
        self.env_config["we"] = [1, 0]

    def objective(self, trial):
        self.args = copy.deepcopy(self.ppo_config)
        self.args.save_dir += "/optuna/" + self.study_name
        self.args.ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1)

        # Set ppo parameters as user attributes.
        for k, v in self.args.__dict__.items():
            trial.set_user_attr(k, v)

        env = gym.make("MultiAgentEthicalGathering-v1", **self.env_config)
        env = NormalizeReward(env)

        # Set environment parameters as user attributes.
        for k, v in self.env_config.items():
            trial.set_user_attr(k, v)

        ppo = ParallelIPPO(self.args, env=env)
        ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
            k: {"actor_lr": trial.suggest_float(f"actor_lr_{k}", 0.00005, 0.001),
                "critic_lr": trial.suggest_float(f"critic_lr_{k}", 0.00005, 0.01)
                } for k in ppo.agents})
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=300),
            TensorBoardLogging(ppo, log_dir="jro/EGG_DATA"),
            AnnealEntropy(ppo, concavity=self.args.concavity_entropy),
        ])

        ppo.train()
        return self.eval_mo(ppo)


if __name__ == "__main__":
    args = args_from_json("hyperparameters/tiny.json")
    args.save_dir += "/optuna"
    opt = Unethical(["maximize", "maximize"], tiny, args, n_trials=1, save=args.save_dir, study_name=args.tag+"_mo")
    opt.optimize()
