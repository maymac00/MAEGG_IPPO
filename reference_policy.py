import copy
import logging
import time

import torch as th
import gym
import numpy as np
import torch.multiprocessing as mp
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO.config import args_from_json


class FindReferencePolicy:
    def __init__(self, env, ppo, warm_up=2, load_from_ckpt=None, **kwargs):
        self.historical_rewards = []
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
            self.warm_up_its = 0
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

        if self.ppo is not None:
            self.ppo._finish_training = _finish_training.__get__(self.ppo)

    def save_data(self):
        """
        Save returns as a csv file. If file already exists rewrite it. The csv file has to be formatted to be read by
        pandas. The first row is the header, the first column is the index which corresponds to iteration.
        The next columns are the returns of each agent followed by some statistics of the returns. The distance metric
        is also saved as the last column.
        :return:
        """
        folder = f"{self.ppo.save_dir}/{self.ppo.tag}"
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Build numpy array
        data = np.zeros((len(self.returns), 2 * self.ppo.n_agents + 1))
        for i, r in enumerate(self.returns):
            data[i, 0] = i
            data[i, 1:1 + self.ppo.n_agents] = r

        data[:, 1 + self.ppo.n_agents] = self.distances

        # Save numpy array
        np.savetxt(f"{folder}/returns.csv", data, delimiter=",")

    def evaluate(self, n_simulations=100):
        agents = self.ppo.agents
        if isinstance(self.env, NormalizeReward):
            self.env.active = False
        for ag in agents.values():
            ag.actor.eval_mode = True
        # Run a simulation of the trained agents

        acc_rewards = np.zeros((n_simulations, len(agents)))
        for sim in range(n_simulations):
            obs, info = self.env.reset()
            done = [False] * len(agents)
            while not all(done):
                actions = [agents[i].actor.predict(obs[i]) for i in range(len(agents))]
                obs, rewards, done, info = self.env.step(actions)
                acc_rewards[sim, :] += rewards
                # env.render()

        if isinstance(self.env, NormalizeReward):
            self.env.active = True
        for ag in agents.values():
            ag.actor.eval_mode = False
        self.historical_rewards.append(acc_rewards)
        return acc_rewards.mean(axis=0)

    def find(self, t_max=10, epsilon=0.15):
        r = self.evaluate(n_simulations=10)
        print(f"r_o : {r}")

        d = np.infty
        self.policies = [{k: v for k, v in self.ppo.agents.items()}]
        self.returns = [r]
        self.distances = [d]
        t = 0
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

            # self.save_data()

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
        p_t, result, id, t = task

        def _finish_training(self):
            pass

        self.ppo._finish_training = _finish_training.__get__(self.ppo)

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
                tasks.append((p_t, d, i, t))

            processes = []
            for i in range(self.ppo.n_agents):
                p = mp.Process(target=self._parallel_training, args=(tasks[i],))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for i in self.ppo.agents.keys():
                self.policies[t][i] = copy.deepcopy(d[i])

    def getPPO(self):
        args = args_from_json("hyperparameters/medium.json")
        args.tot_steps = 10000
        ppo = IPPO(args, env=env)
        ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
        ppo.addCallbacks(PrintAverageReward(ppo, 1), private=True)
        ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, args.concavity_entropy), private=True)
        return ppo


if __name__ == "__main__":
    from EthicalGatheringGame import MAEGG
    from EthicalGatheringGame.presets import tiny, medium
    from EthicalGatheringGame.wrappers import NormalizeReward
    from IndependentPPO import IPPO, ParallelIPPO
    from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging
    from IndependentPPO.lr_schedules import DefaultPPOAnnealing
    from IndependentPPO.config import args_from_json

    medium["we"] = [1, 99]
    env = gym.make("MultiAgentEthicalGathering-v1", **medium)
    env = NormalizeReward(env)
    args = args_from_json("hyperparameters/medium.json")
    args.tot_steps = 30000
    outerppo = IPPO(args, env=env)
    outerppo.lr_scheduler = DefaultPPOAnnealing(outerppo)
    outerppo.addCallbacks(PrintAverageReward(outerppo, 1), private=True)
    outerppo.addCallbacks(AnnealEntropy(outerppo, 1.0, 0.5, args.concavity_entropy), private=True)

    finder = ParallelFindReferencePolicy(env, outerppo)
    finder.find()
