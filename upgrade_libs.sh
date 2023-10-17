#!/bin/bash
source /home/arnau/tools/python_venv/MAEGG_IPPO/bin/activate
pip uninstall multiagentethicalgathering
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
pip uninstall ippo
pip install git+https://github.com/maymac00/Independent_PPO.git