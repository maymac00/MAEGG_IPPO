from EthicalGatheringGame.presets import medium
from hyper_tuning import OptimizerMAEGG, OptimizerMOMAGG
from IndependentPPO.config import args_from_json


if __name__ == "__main__":
    args = args_from_json("hyperparameters/medium.json")
    args.save_dir += "/optuna"
    medium["we"] = [1, 99]
    opt = OptimizerMAEGG("maximize", medium, args, n_trials=1, save=args.save_dir, study_name=args.tag)
    opt.optimize()
