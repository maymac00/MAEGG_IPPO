from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *

import gym

env = gym.make("MultiAgentEthicalGathering-v1", **medium)

agents = IPPO.agents_from_file("jro/EGG_DATAethical_medium_3/ethical_medium_3/2500_200000_1")
SoftmaxActor.action_selection = no_filter
env.setTrack(True)
env.setStash(True)
env.reset()
for r in range(100):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        env.render(mode="partial_observability")
env.plot_results("median")
env.print_results()
