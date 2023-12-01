from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO.IPPO import IPPO
from IndependentPPO.config import args_from_json
from IndependentPPO.callbacks import *
from hyper_tuning import OptimizerMOMAGG
import gym


if __name__ == "__main__":
    args = args_from_json("hyperparameters/tiny.json")
    args.save_dir += "/optuna"
    opt = OptimizerMOMAGG(["maximize", "maximize"], tiny, args, n_trials=1, save=args.save_dir, study_name=args.tag + "_mo")
    opt.optimize()
