import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
import gym

large["we"] = [1, 0.91]
env = gym.make("MultiAgentEthicalGathering-v1", **large)
# env = NormalizeReward(env)

agents = IPPO.actors_from_file("EGG_DATA/ethical_large_we10_try2/ethical_large_we10_try2/2500_80000_1_(29)")
env.setTrack(True)
env.setStash(True)
env.reset()
history = []
mo_history = []
for r in range(1000):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        # env.render(mode="partial_observability")
    print(f"Epsiode {r}: {acc_reward} \t Agents (V_0, V_e): ", "\t\t".join([f"{np.round(env.agents[i].r_vec)}" for i in range(env.n_agents)]))
    mo_history.append([env.agents[i].r_vec for i in range(env.n_agents)])
    history.append(acc_reward)

mo_history = np.array(mo_history)
history = np.array(history)
# Print history mean
print(f"\nMean reward per agent: {list(history.mean(axis=0))}")
print(f"\nMean mo reward per agent: {list(mo_history.mean(axis=0))}")
env.plot_results("median")
env.print_results()
