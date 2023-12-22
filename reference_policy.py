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
    def __init__(self, env, ppo, load_from_ckpt=None):
        self.env = env
        self.ppo = ppo
        self.policies = []
        self.returns = []
        self.p_t = None

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
        self.policies = []
        self.returns = []
        self.p_t = None
        r = self.evaluate(n_simulations=10)
        print(f"r_o : {r}")
        epsilon = 0.1

        d = np.infty
        self.policies = [{k: v for k, v in self.ppo.agents.items()}]
        self.p_t = self.policies[0]
        self.returns = [r]
        t = 0
        t_max = 10
        while d >= epsilon and t < t_max:
            t += 1
            self.logger.info(f"==============================")
            self.logger.info(f"Computing best response for t = {t}")
            time0 = time.time()
            self.compute_best_response(t)

            # Set current state to the last joint policy
            self.ppo.agents = self.policies[t]

            d = self.compute_distance_metric()
            self.logger.info(f"Return of pi_t : {self.returns[t]}")
            self.logger.info(f"Distance metric: {d}")
            self.logger.info(f"Time elapsed: {time.time() - time0} s")
        return self.policies[t], self.returns[t]

    def compute_best_response(self, t):
        self.policies.append({k: None for k in self.ppo.r_agents})
        for i in self.ppo.agents.keys():
            self.p_t = copy.deepcopy(self.policies[t - 1])
            for aux in self.p_t.values():
                aux.freeze()
            self.p_t[i].unfreeze()
            self.ppo.train(set_agents=self.p_t)
            self.policies[t][i] = copy.deepcopy(self.ppo.agents[i])

    def compute_distance_metric(self):
        r = self.evaluate(n_simulations=100)
        d = np.linalg.norm(self.returns[-1] - r)
        self.returns.append(r)
        return d


class ParallelFindReferencePolicy(FindReferencePolicy):
    def __init__(self, env, ppo, load_from_ckpt=None):
        super().__init__(env, ppo, load_from_ckpt)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    def _parallel_training(self, task):
        th.set_num_threads(10)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        p_t, result, id = task
        self.ppo.train(set_agents=p_t)
        result[id] = self.ppo.agents[id]

    def compute_best_response(self, t):
        self.policies.append({k: None for k in self.ppo.r_agents})
        tasks = []
        with mp.Manager() as manager:
            d = manager.dict()
            for i in self.ppo.agents.keys():
                self.p_t = copy.deepcopy(self.policies[t - 1])
                for aux in self.p_t.values():
                    aux.freeze()
                self.p_t[i].unfreeze()
                tasks.append((self.p_t, d, i))

            processes = []
            for i in range(self.ppo.n_agents):
                p = mp.Process(target=self._parallel_training, args=(tasks[i],))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for i in self.ppo.agents.keys():
                self.policies[t][i] = d[i]


class FindOptimalReferencePolicy(FindReferencePolicy):
    def __init__(self, env, ppo, load_from_ckpt=None):
        super().__init__(env, ppo, load_from_ckpt)
        # We create a new study object for each agent
        self.agent_studies = []
        for i in range(self.ppo.n_agents):
            try:
                s = optuna.load_study(
                    study_name=f'agent_{i}',
                    storage=f'sqlite:///{self.ppo.save_dir}/optuna_agent_{i}.db'
                )
                print(f"Loaded existing study 'agent_{i}' with {len(s.trials)} trials.")
            except optuna.exceptions.StorageInternalError:
                # Make sure path exists
                os.makedirs(self.ppo.save_dir, exist_ok=True)
                # Create file
                f = open(f"{self.ppo.save_dir}/optuna_agent_{i}.db", "w+")
                # close file
                f.close()
                s = optuna.create_study(
                    direction='maximize',
                    sampler=DecreasingCandidatesTPESampler(),
                    study_name=f'agent_{i}',
                    pruner=optuna.pruners.NopPruner(),
                    storage=f'sqlite:///{self.ppo.save_dir}/optuna_agent_{i}.db'
                )
            self.agent_studies.append(s)

        self.active_agent = 0

    def objective(self, trial):
        self.ppo.ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, step=0.0005)
        self.ppo.actor_lr = trial.suggest_float("actor_lr", 0.000005, 0.001, step=0.000005)
        self.ppo.critic_lr = trial.suggest_float("critic_lr", 0.00005, 0.01, step=0.00005)

        self.ppo.train(set_agents=self.p_t)

        return self.evaluate(self.ppo, n=50)[self.active_agent]

    def compute_best_response(self, t):
        self.policies.append({k: None for k in self.ppo.r_agents})
        for i in self.ppo.agents.keys():
            self.active_agent = i
            self.p_t = copy.deepcopy(self.policies[t - 1])
            for aux in self.p_t.values():
                aux.freeze()
            self.p_t[i].unfreeze()
            self.agent_studies[i].optimize(self.objective, n_trials=1)
            self.policies[t][i] = copy.deepcopy(self.ppo.agents[i])


def main1():
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    env = NormalizeReward(env)
    args = args_from_json("/home/arnau/PycharmProjects/MAEGG_IPPO/hyperparameters/tiny.json")
    args.tot_steps = 15000
    ppo = ParallelIPPO(args, env=env)

    ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
    ppo.addCallbacks([
        PrintAverageReward(ppo, n=150),
        AnnealEntropy(ppo, concavity=args.concavity_entropy),
    ])

    opt = FindOptimalReferencePolicy(env, ppo)
    opt.find()

def main2():
    tiny["we"] = [1, 99]
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    env = NormalizeReward(env)
    args = args_from_json("hyperparameters/tiny.json")
    args.tot_steps = 30000
    ppo = ParallelIPPO(args, env=env)
    ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
    ppo.addCallbacks(PrintAverageReward(ppo, 1000))
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, args.concavity_entropy))
    ppo.addCallbacks(SaveCheckpoint(ppo))

    finder = ParallelFindReferencePolicy(env, ppo)
    finder.find()


if __name__ == "__main__":
    main2()