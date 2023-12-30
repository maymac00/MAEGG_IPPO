import copy

from reference_policy import ParallelFindReferencePolicy
from IndependentPPO.hypertuning import DecreasingCandidatesTPESampler
from IndependentPPO import IPPO
import torch as th
import optuna
import os


# TODO: Fix how i save the ppo parameters into config.json. I should save params for each agent. Now config.json is
#  useless for AutoTunedFRP
class AutoTunedFRP(ParallelFindReferencePolicy):
    def __init__(self, env, ppo, warm_up=5, load_from_ckpt=None, initial_candidates=12, **kwargs):
        super().__init__(env, ppo, warm_up=warm_up, load_from_ckpt=load_from_ckpt, **kwargs)
        os.makedirs(f"{self.ppo.init_args.save_dir}/{self.ppo.init_args.tag}", exist_ok=True)
        self.study = {
            k: optuna.create_study(
                direction="maximize",
                study_name=self.ppo.init_args.tag + "_" + str(k),
                storage=f"sqlite:///{self.ppo.init_args.save_dir}/{self.ppo.init_args.tag}/autotune_reference_policy_{k}.db",
                load_if_exists=True,
                sampler=DecreasingCandidatesTPESampler(initial_n_ei_candidates=initial_candidates),
            ) for k in self.ppo.r_agents}

    def _parallel_training(self, task):
        """
        Custom optuna loop for this class that runs just one trial
        :param i: non-frozen agent key
        :return:
        """
        th.set_num_threads(1)

        def _finish_training(self):
            pass

        self.ppo._finish_training = _finish_training.__get__(self.ppo)

        p_t, result, i = task
        # Pre trial
        pass

        # Create trial
        trial = self.study[i].ask()

        # Pre objective. Here we change PPO's parameters
        self.ppo.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
        self.ppo.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
        self.ppo.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)

        # Run objective
        try:
            self.ppo.train(set_agents=p_t)
            result[i] = self.ppo.agents[i]

        except Exception as e:
            raise e

        # If trial number < than warm up, we set optuna trial to fail
        if trial.number < self.warm_up_its:
            self.study[i].tell(trial, state=optuna.trial.TrialState.FAIL)
            return None

        acc_return = self.evaluate(n_simulations=100)
        value = acc_return[i]

        trial.set_user_attr("acc_return_per_agent", list(acc_return))
        for k, v in self.ppo.__dict__.items():
            try:
                trial.set_user_attr(k, v)
            except TypeError as e:
                continue

        self.study[i].tell(trial, value)
        print(
            f"Trial for agent {i} completed with value {value}.")
        return value


if __name__ == "__main__":
    from EthicalGatheringGame import MAEGG
    from EthicalGatheringGame.presets import tiny, small, medium, large
    from EthicalGatheringGame.wrappers import NormalizeReward
    from IndependentPPO import IPPO
    from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging, SaveCheckpoint
    from IndependentPPO.lr_schedules import DefaultPPOAnnealing, IndependentPPOAnnealing
    from IndependentPPO.config import args_from_json

    tiny["we"] = [1, 99]
    env = MAEGG(**tiny)
    env = NormalizeReward(env)

    args = args_from_json("hyperparameters/tiny.json")
    args.tot_steps = 5000
    ppo = IPPO(args, env=env)
    ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
    ppo.addCallbacks(PrintAverageReward(ppo, 5))
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, args.concavity_entropy))

    atfrp = AutoTunedFRP(env, ppo)
    atfrp.find()
