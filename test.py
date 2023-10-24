from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny
from IndependentPPO import IPPO
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

agents = IPPO.agents_from_file("EGG_DATA/tiny/2500_30000_1")

env.track = True
env.stash_runs = True
env.reset()
print("1234")
for r in range(10):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        print(info)
        # env.render()
env.plot_results("line")