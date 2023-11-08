from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small
from EthicalGatheringGame.wrappers import DummyPeers
from IndependentPPO import IPPO
import gym
from IndependentPPO.config import args_from_json

small["we"] = [1, 999]
print("Dummy testing")
env = gym.make("MultiAgentEthicalGathering-v1", **small)
dummy_mask = [1, 0]
env = DummyPeers(env, dummy_mask)

args = args_from_json("hyperparameters/small.json")
ppo = IPPO(args, env=env)
print(args)
ppo.train()