import numpy as np
import optuna
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO import IPPO, ParallelIPPO
from IndependentPPO.lr_schedules import DefaultPPOAnnealing, IndependentPPOAnnealing
from IndependentPPO.config import args_from_json
from IndependentPPO.callbacks import PrintAverageReward, TensorBoardLogging, AnnealEntropy
import gym


def objective(trial):
    args = args_from_json("hyperparameters/medium.json")
    args.save_dir += "/optuna/" + args.tag
    medium["we"] = [1, 99]
    env = gym.make("MultiAgentEthicalGathering-v1", **medium)
    env = NormalizeReward(env)

    for k, v in tiny.items():
        trial.set_user_attr(k, v)

    args.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.01)
    ppo = ParallelIPPO(args, env=env)
    ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
        0: {
            "actor_lr": trial.suggest_float("actor_lr_0", 0.00005, 0.001),
            "critic_lr": trial.suggest_float("critic_lr_0", 0.00005, 0.01)},
        1: {
            "actor_lr": trial.suggest_float("actor_lr_1", 0.00005, 0.001),
            "critic_lr": trial.suggest_float("critic_lr_1", 0.00005, 0.01)},
        2: {
            "actor_lr": trial.suggest_float("actor_lr_2", 0.00005, 0.001),
            "critic_lr": trial.suggest_float("critic_lr_2", 0.00005, 0.01)},
    })
    ppo.addCallbacks([
        PrintAverageReward(ppo, n=500),
        TensorBoardLogging(ppo, log_dir="jro/EGG_DATA/optuna"),
        AnnealEntropy(ppo, concavity=args.concavity_entropy),
    ])
    trial.set_user_attr("run_name", ppo.run_name)
    ppo.train()
    trial.set_user_attr("save_dir", ppo.folder)
    metric = np.zeros(ppo.n_agents)
    ppo.eval_mode = True
    for i in range(20):  # Rollout does n_steps so keep it low
        rec = ppo.rollout()
        metric += rec
    metric /= 20
    return metric


if __name__ == "__main__":
    args = args_from_json("hyperparameters/tiny.json")
    save = args.save_dir + "/optuna/" + args.tag
    storage_path = f'sqlite:///{save}/database.db'
    study_name = args.tag

    try:
        # Try to load the existing study.
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        print(f"Loaded existing study '{study_name}' with {len(study.trials)} trials.")
    except:
        # If the study does not exist, create a new one.
        import os

        os.makedirs(save, exist_ok=True)
        # Create file
        f = open(f"{save}/database.db", "w+")
        # close file
        f.close()
        study = optuna.create_study(direction=["maximize"]*args.n_agents, study_name=study_name, storage=storage_path)
        print(f"Created new study '{study_name}'.")

    study.optimize(objective, n_trials=1)
