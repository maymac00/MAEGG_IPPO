import optuna
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO.callbacks import LearningRateDecay, AnnealEntropy, PrintAverageReward, TensorBoardLogging

from IndependentPPO.IPPO import IPPO
from IndependentPPO.config import args_from_json
import gym
import matplotlib

args = args_from_json("hyperparameters/tiny.json")


def objective(trial):
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    args.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
    args.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
    args.tot_steps = trial.suggest_int("tot_steps", 15000000, 25000000, step=5000000)
    # args.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)
    args.concavity_entropy = trial.suggest_float("concavity-entropy", 1.0, 3.5)
    ppo = IPPO(args, env=env)
    ppo.addCallbacks([
        LearningRateDecay(ppo),
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
        rec = ppo._sim()
        metric += sum(rec["reward_per_agent"]) / args.n_agents
    metric /= 20
    return metric


if __name__ == "__main__":
    args.save_dir += "/optuna/" + args.tag
    storage_path = f'sqlite:///{args.save_dir}/database.db'
    study_name = args.tag

    try:
        # Try to load the existing study.
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        print(f"Loaded existing study '{study_name}' with {len(study.trials)} trials.")
    except:
        # If the study does not exist, create a new one.
        import os

        os.makedirs(args.save_dir, exist_ok=True)
        # Create file
        f = open(f"{args.save_dir}/database.db", "w+")
        # close file
        f.close()
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_path)
        print(f"Created new study '{study_name}'.")

    study.optimize(objective, n_trials=2)
