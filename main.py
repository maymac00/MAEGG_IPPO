from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward
from IndependentPPO import IPPO, ParallelIPPO
from IndependentPPO.config import args_from_json
from IndependentPPO.callbacks import *
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
env = NormalizeReward(env)
args = args_from_json("hyperparameters/tiny.json")
ppo = ParallelIPPO(args, env=env)
ppo.addCallbacks([
    PrintAverageReward(ppo, n=300),
    TensorBoardLogging(ppo, log_dir="jro/EGG_DATA"),
    AnnealEntropy(ppo, concavity=args.concavity_entropy),
])
print(args)
ppo.train()
