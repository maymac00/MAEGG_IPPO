import copy
import time
import gym
import numpy as np
from EthicalGatheringGame import NormalizeReward
from EthicalGatheringGame.presets import large
from IndependentPPO import ParallelIPPO, args_from_json
from IndependentPPO.callbacks import TensorBoardLogging, SaveCheckpoint, AnnealEntropy, \
    PrintAverageReward
from IndependentPPO.lr_schedules import IndependentPPOAnnealing
import argparse


args = args_from_json("hyperparameters/large.json")

conv_args = argparse.ArgumentParser()

conv_args.add_argument("--eff-rate", type=float, default=0.2)
conv_args.add_argument("--db", type=int, default=10)
conv_args.add_argument("--we", type=float, default=10)

conv_args.add_argument("--actor-lr0", type=float, default=1e-04)
conv_args.add_argument("--critic-lr0", type=float, default=100e-04)
conv_args.add_argument("--actor-lr1", type=float, default=1e-04)
conv_args.add_argument("--critic-lr1", type=float, default=100e-04)
conv_args.add_argument("--eval-sims", type=int, default=500)

conv_args , unknown = conv_args.parse_known_args()
# Log every argument
for arg in vars(conv_args):
    print(f"{arg}: {getattr(conv_args, arg)}")

large["efficiency"] = [0.85] * int(args.n_agents * conv_args.eff_rate) + [0.2] * int(args.n_agents - conv_args.eff_rate * args.n_agents)
large["donation_capacity"] = conv_args.db
large["color_by_efficiency"] = True
large["objective_order"] = "individual_first"
large["we"] = [1, conv_args.we]

env = gym.make("MultiAgentEthicalGathering-v1", **large)
env = NormalizeReward(env)

final_value = 0.4

# We make groups of efficiency to reduce the amount of parameters to tune.
eff_dict = {}
eff_groups = [np.where(env.efficiency == value)[0].tolist() for value in np.unique(env.efficiency)]
eff_dict = {}
for k in eff_groups[0]:
    eff_dict[k] = {
        "actor_lr": conv_args.actor_lr0,
        "critic_lr": conv_args.critic_lr0
    }
for k in eff_groups[1]:
    eff_dict[k] = {
        "actor_lr": conv_args.actor_lr1,
        "critic_lr": conv_args.critic_lr1
    }

# Do not modify args after this point.
ppo = ParallelIPPO(args, env=env)
ppo.lr_scheduler = IndependentPPOAnnealing(ppo, eff_dict)
ppo.addCallbacks([
    PrintAverageReward(ppo, n=5000, show_time=True),
    TensorBoardLogging(ppo, log_dir=f"{args.save_dir}/{args.tag}/log/{ppo.run_name}", f=100),
    SaveCheckpoint(ppo, 5000),
    AnnealEntropy(ppo, final_value=final_value, concavity=args.concavity_entropy),
])

ppo.train()

# Do 500 rollouts
actors = [ag.actor for ag in ppo.agents.values()]
for ag in actors:
    ag.eval_mode = True

env.toggleTrack(True)
env.toggleStash(True)
env.reset()
aux_cont = [0] * env.n_agents
aux_cost = [0] * env.n_agents
last_apple_value = [0] * env.n_agents
history = np.zeros((conv_args.eval_sims, env.n_agents))
mo_history = np.zeros((conv_args.eval_sims, env.n_agents, 2))
for r in range(conv_args.eval_sims):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    acc_reward_mo = np.zeros((env.n_agents, 2))
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(actors)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        acc_reward_mo += np.array([ag.r_vec for ag in env.agents.values()])

    mo_history[r] = acc_reward_mo
    history[r] = acc_reward

# Print history mean
print(f"\nMean reward per agent: {list(history.mean(axis=0))}")
print(f"\nMean mo reward per agent: {list(mo_history.mean(axis=0))}")
env.plot_results("median", ppo.folder + "/median_plot.png", show=False)
env.print_results()