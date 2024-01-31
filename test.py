from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
import gym

medium["we"] = [1, 0]
env = gym.make("MultiAgentEthicalGathering-v1", **medium)
# env = NormalizeReward(env)

agents = IPPO.actors_from_file("jro/EGG_DATA/ethical_medium_we0_try2/ethical_medium_we0_try2/2500_60000_1_(28)")
env.setTrack(True)
env.setStash(True)
env.reset()
history = []
mo_history = []
for r in range(100):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        # env.render(mode="partial_observability")
    print(f"Epsiode {r}: {acc_reward} \t Agents (V_0, V_e):\t ({env.agents[0].r_vec}), \t({env.agents[1].r_vec}), \t({env.agents[2].r_vec})")
    mo_history.append([env.agents[0].r_vec, env.agents[1].r_vec, env.agents[2].r_vec])
    history.append(acc_reward)

mo_history = np.array(mo_history)
history = np.array(history)
# Print history mean
print(f"\nMean reward per agent: {list(history.mean(axis=0))}")
print(f"\nMean mo reward per agent: {list(mo_history.mean(axis=0))}")
env.plot_results("median")
env.print_results()
