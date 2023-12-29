import copy
import logging
import time

from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO import ParallelIPPO, IPPO
from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging, SaveCheckpoint
from IndependentPPO.lr_schedules import DefaultPPOAnnealing, IndependentPPOAnnealing
from IndependentPPO.config import args_from_json
import torch as th
import gym
import matplotlib
import numpy as np
import torch.multiprocessing as mp


class FindReferencePolicy:
    def __init__(self, env, ppo, warm_up=5, load_from_ckpt=None, **kwargs):
        self.env = env
        self.ppo = ppo
        self.policies = []
        self.returns = []
        self.distances = []
        self.warm_up_its = warm_up

        # Handle kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        if load_from_ckpt is not None:
            self.ppo.load_checkpoint(load_from_ckpt)

        self.logger = logging.getLogger("FindReferencePolicy")
        self.logger.setLevel(logging.INFO)
        if len(self.logger.handlers) == 0:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Morph ppo to not save anything with monkey patching
        def _finish_training(self):
            pass

        self.ppo._finish_training = _finish_training.__get__(self.ppo)

    def evaluate(self, n_simulations=100):
        agents = self.ppo.agents
        if isinstance(self.env, NormalizeReward):
            self.env.active = False
        for ag in agents.values():
            ag.actor.eval_mode = True
        # Run a simulation of the trained agents

        acc_reward = np.zeros(len(agents))
        for sim in range(n_simulations):
            obs, info = self.env.reset()
            done = [False] * len(agents)
            while not all(done):
                actions = [agents[i].actor.predict(obs[i]) for i in range(len(agents))]
                obs, rewards, done, info = self.env.step(actions)
                acc_reward += rewards
                # env.render()

        if isinstance(self.env, NormalizeReward):
            self.env.active = True
        for ag in agents.values():
            ag.actor.eval_mode = False

        return acc_reward / n_simulations

    def find(self):
        r = self.evaluate(n_simulations=10)
        print(f"r_o : {r}")
        epsilon = 0.1

        d = np.infty
        self.policies = [{k: v for k, v in self.ppo.agents.items()}]
        self.returns = [r]
        self.distances = [d]
        t = 0
        t_max = 10
        while not self.stop_condition(t, epsilon, t_max):
            t += 1
            self.logger.info(f"==============================")
            self.logger.info(f"Computing best response for t = {t}")

            time0 = time.time()
            self.compute_best_response(t)
            timespan = time.time() - time0

            # Set current state to the last joint policy
            self.ppo.agents = self.policies[t]

            self.ppo.save_experiment_data(folder=f"{self.ppo.save_dir}/{self.ppo.tag}/AT_RP_it_{t}")

            self.compute_metrics(t)  # Compute returns and distances

            self.logger.info(f"Return of pi_t : {self.returns[t]}")
            self.logger.info(f"Distance metric: {self.distances[t]}")

            # Format time elapsed to HH:MM:SS
            self.logger.info(
                f"Time elapsed (without sim time): {int(timespan // 3600)}:{int(timespan // 60) % 60}:{int(timespan % 60)}")
        return self.policies[t], self.returns[t]

    def compute_metrics(self, t):
        """
        Update the state of the algorithm. By this function, all agent policies of ppo has been set to pi_t.
        self.ppo.agents = self.policies[t]
        But returns and distances have not been calculated yet.
        :param t:
        :return:
        """
        r = self.evaluate(n_simulations=10) if t < self.warm_up_its else self.evaluate(n_simulations=100)
        d = self.compute_distance_metric(r)
        self.returns.append(r)
        self.distances.append(d)

    def stop_condition(self, t, epsilon, t_max):
        """
        Stop condition for the algorithm. Returns True if the algorithm should stop, False otherwise.
        :param t:
        :param epsilon:
        :param t_max:
        :return:
        """
        if t < self.warm_up_its:
            return False
        return t >= t_max or self.distances[t] < epsilon

    def compute_best_response(self, t):
        self.policies.append({k: None for k in self.ppo.r_agents})
        for i in self.ppo.agents.keys():
            p_t = copy.deepcopy(self.policies[t - 1])
            for aux in p_t.values():
                aux.freeze()
            p_t[i].unfreeze()
            self.ppo.train(set_agents=p_t)
            self.policies[t][i] = copy.deepcopy(self.ppo.agents[i])

    def compute_distance_metric(self, r):
        d = np.linalg.norm(self.returns[-1] - r)
        return d


class ParallelFindReferencePolicy(FindReferencePolicy):
    def __init__(self, env, ppo, warm_up=5, load_from_ckpt=None, **kwargs):
        super().__init__(env, ppo, warm_up, load_from_ckpt, **kwargs)
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            pass

    def _parallel_training(self, task):
        th.set_num_threads(1)

        def _finish_training(self):
            pass

        self.ppo._finish_training = _finish_training.__get__(self.ppo)

        p_t, result, id = task
        self.ppo.train(set_agents=p_t)
        result[id] = self.ppo.agents[id]

    def compute_best_response(self, t):
        th.set_num_threads(1)
        self.policies.append({k: None for k in self.ppo.r_agents})
        tasks = []
        with mp.Manager() as manager:
            d = manager.dict()
            for i in self.ppo.agents.keys():
                p_t = copy.deepcopy(self.policies[t - 1])
                for aux in p_t.values():
                    aux.freeze()
                p_t[i].unfreeze()
                tasks.append((p_t, d, i))

            processes = []
            for i in range(self.ppo.n_agents):
                p = mp.Process(target=self._parallel_training, args=(tasks[i],))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for i in self.ppo.agents.keys():
                self.policies[t][i] = copy.deepcopy(d[i])


if __name__ == "__main__":
    tiny["we"] = [1, 99]
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    env = NormalizeReward(env)
    args = args_from_json("hyperparameters/tiny.json")
    args.tot_steps = 30000
    ppo = IPPO(args, env=env)
    ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
    ppo.addCallbacks(PrintAverageReward(ppo, 1))
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, args.concavity_entropy))

    finder = ParallelFindReferencePolicy(env, ppo)
    finder.find()
