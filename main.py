from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO.IPPO import IPPO
from IndependentPPO.config import args_from_json
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
args = args_from_json("hyperparameters/new_tiny.json")
ppo = IPPO(args, env=env)
print(args)
ppo.train()
