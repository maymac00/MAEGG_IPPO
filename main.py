"""
Install MA ethical gathering game as test environment and its dependencies
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
"""
import numpy as np
from EthicalGatheringGame.presets import tiny, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO import ParallelIPPO
from IndependentPPO.IPPO import IPPO
from IndependentPPO.CIPPO import CIPPO, ParallelCIPPO
from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, AnnealActionFilter, TensorBoardLogging, \
    SaveCheckpoint
from IndependentPPO.config import args_from_json
from IndependentPPO.lr_schedules import IndependentPPOAnnealing
import gym
import matplotlib


large["we"] = [1, 0]
large["donation_capacity"] = 0


args = args_from_json("hyperparameters/large.json")
args.n_epochs = 20
args.n_cpus = 10

eff_rate = 0.8

large["efficiency"] = [0.85] * int(args.n_agents * eff_rate) + [0.2] * int(args.n_agents - eff_rate * args.n_agents)
print(large["efficiency"])
env = gym.make("MultiAgentEthicalGathering-v1", **large)
env = NormalizeReward(env)


eff_groups = [np.where(env.efficiency == value)[0].tolist() for value in np.unique(env.efficiency)]
eff_dict = {}
for k in eff_groups[0]:
    eff_dict[k] = {"actor_lr": 6e-03, "critic_lr": 300e-03}
for k in eff_groups[1]:
    eff_dict[k] = {"actor_lr": 6e-03, "critic_lr": 300e-03}
"""
for i, group in enumerate(eff_groups):
    actorlr = trial.suggest_float(f"actor_lr_{i}", 0.000001, 0.00001, step=0.000005)
    criticlr = trial.suggest_float(f"critic_lr_{i}", 0.0002, 0.001, step=0.0001)
    for agent in group:
        eff_dict[agent] = {"actor_lr": actorlr, "critic_lr": criticlr}
"""

ppo = ParallelIPPO(args, env)
ppo.lr_scheduler = IndependentPPOAnnealing(ppo, eff_dict)

ppo.addCallbacks([
    PrintAverageReward(ppo, n=5000, show_time=True),
    TensorBoardLogging(ppo, log_dir=f"{args.save_dir}/{args.tag}/log/{ppo.run_name}", f=10),
    SaveCheckpoint(ppo, 5000),
    AnnealEntropy(ppo, final_value=0.4, concavity=2),
])
if args.anneal_action_filter:
    ppo.addCallbacks(AnnealActionFilter(ppo))

ppo.train()
