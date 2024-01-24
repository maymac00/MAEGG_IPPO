from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
import gym

medium["we"] = [1, 10]
env = gym.make("MultiAgentEthicalGathering-v1", **medium)
# env = NormalizeReward(env)

agents = IPPO.actors_from_file("jro/EGG_DATA/autotuned_reference_policy_try1/AT_RP_it_6")
env.setTrack(True)
env.setStash(True)
env.reset()
for r in range(100):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        # env.render(mode="partial_observability")
    print(f"Epsiode {r}: {acc_reward}")
env.plot_results("median")
env.print_results()
