from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *

import gym

tiny["we"] = [1, 99]
env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
# env = NormalizeReward(env)

agents = IPPO.actors_from_file("EGG_DATA/reference_policy/2500_30000_1_(15)")
SoftmaxActor.action_selection = no_filter
env.setTrack(True)
env.setStash(True)
env.reset()
for r in range(10):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        # env.render()
    print(f"Epsiode {r}: {acc_reward}")
env.plot_results("median")
env.print_results()
