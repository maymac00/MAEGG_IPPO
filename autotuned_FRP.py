import os
import time
from collections import deque

import numpy as np
from scipy.stats import ttest_rel, ks_2samp
from torch.utils.tensorboard import SummaryWriter
import optuna
import torch as th
from IndependentPPO.hypertuning import DecreasingCandidatesTPESampler

from reference_policy import ParallelFindReferencePolicy


# TODO: Fix how i save the ppo parameters into config.json. I should save params for each agent. Now config.json is
#  useless for AutoTunedFRP
class AutoTunedFRP(ParallelFindReferencePolicy):
    def __init__(self, env, ppo, tmax=10, warm_up=5, load_from_ckpt=None, initial_candidates=12, **kwargs):
        super().__init__(env, ppo, tmax=tmax, warm_up=warm_up, load_from_ckpt=load_from_ckpt, **kwargs)
        os.makedirs(f"{self.ppo.init_args.save_dir}/{self.ppo.init_args.tag}", exist_ok=True)
        self.study = {
            k: optuna.create_study(
                direction="maximize",
                study_name=self.ppo.init_args.tag + "_" + str(k),
                storage=f"sqlite:///{self.ppo.init_args.save_dir}/{self.ppo.init_args.tag}/autotuned_reference_policy_{k}.db",
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

        # self.ppo.addCallbacks(TensorBoardLogging(ppo, log_dir=f"{args.save_dir}/{args.tag}/log", f=3))

        def _finish_training(self):
            pass

        self.ppo._finish_training = _finish_training.__get__(self.ppo)

        # self.ppo.callbacks.append(log)

        p_t, result, i, t = task
        # Pre trial
        pass

        # Create trial
        trial = self.study[i].ask()

        # Pre objective. Here we change PPO's parameters
        self.ppo.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001)
        self.ppo.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01)
        self.ppo.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)

        aux = self.ppo.run_name
        self.ppo.run_name = self.ppo.run_name + f"it_{t}_ag_" + str(i)

        # Run objective
        try:
            self.ppo.train(set_agents=p_t)
            result[i] = self.ppo.agents[i]
        except Exception as e:
            print("Exception occurred in train.")
            print(e)
            raise e

        self.ppo.run_name = aux

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

    def compute_distance_metric(self, r):
        """
        Compute distance metric for each agent. In this case we make a paired-sample t-test between the vector of values
        in all the simulations of the t iteration and the vector of values in all the simulations of the t-1 iteration
        (for each agent)
        :param r:
        :return:
        """
        if len(self.historical_rewards) < 2:
            return np.infty

        if self.historical_rewards[-1][:, 0].shape != self.historical_rewards[-2][:, 0].shape:
            return np.infty

        ks = -np.infty
        for ag in range(self.ppo.n_agents):
            kstat, p_value = ks_2samp(self.historical_rewards[-1][:, ag], self.historical_rewards[-2][:, ag])
            ks = max(ks, kstat)
        self.logger.info(f"Kolmogorov-Smirnov statistic: {ks}. Should we stop?: {ks < 0.05}")

        # Calc t-test
        t_test = -np.infty
        p_value = np.infty
        for ag in range(self.ppo.n_agents):
            t, p = ttest_rel(self.historical_rewards[-1][:, ag], self.historical_rewards[-2][:, ag])
            p_value = min(p_value, p)
            t_test = max(t_test, t)
        self.logger.info(f"t-test statistic: {t_test}")
        self.logger.info(f"p-value: {p_value}. Should we stop?: {p_value > 0.05}")

        return p_value

    def stop_condition(self, t, epsilon, t_max):
        if t >= t_max:
            return True
        if t < self.warm_up_its:
            return False
        if self.distances[-1] == np.infty or self.distances[-1] == -np.infty:
            return False
        return self.distances[-1] > epsilon


if __name__ == "__main__":
    from EthicalGatheringGame import MAEGG
    from EthicalGatheringGame.presets import tiny, medium
    from EthicalGatheringGame.wrappers import NormalizeReward
    from IndependentPPO import IPPO, ParallelIPPO
    from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging
    from IndependentPPO.lr_schedules import DefaultPPOAnnealing
    from IndependentPPO.config import args_from_json

    preset = medium
    preset["we"] = [1, 10]
    env = MAEGG(**preset)
    env = NormalizeReward(env)

    args = args_from_json("hyperparameters/medium.json")
    ppo = IPPO(args, env=env)
    ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
    printer = PrintAverageReward(ppo, 5000)
    ppo.addCallbacks(printer)
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.3, args.concavity_entropy))

    atfrp = AutoTunedFRP(env, ppo)
    atfrp.find()
