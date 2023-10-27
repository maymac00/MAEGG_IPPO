#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem 1G
#SBATCH --time=6:00:00
#SBATCH --job-name="Gathering"
# export PATH=/mnt/beegfs/iiia/arnau_mayoral/conda/envs/Gathering/bin:~/conda/bin:$PATH
# export PATH=$STORE/conda/envs/framework/bin:$STORE/conda/bin:$PATH

export PATH=$STORE/conda/envs/framework/bin:$STORE/conda/bin:$PATH
source activate MAEGG_IPPO
export PYTHONUNBUFFERED=TRUE

# srun python ippo.py --tag big_map_first_tries --seed 1 --tot-steps 5000000 --early-stop 2500000 --actor-lr 0.003 --critic-lr 0.01 --map-size big --n-agents 5 --survival-threshold 50 --donation-capacity 25 --apple-regen 0.002
# srun python run_policy_PPO.py

./upgrade_libs.sh > /dev/null

echo "Running job $SLURM_JOB_ID"
echo "Code used for IPPO:"
curl -s "https://api.github.com/repos/maymac00/Independent_PPO/commits" | grep -E -m 1 '"sha"|"date"' | sed -E 's/.*: "(.*)".*/\1/'
echo "Code used for MAEGathering:"
curl -s "https://api.github.com/repos/maymac00/MultiAgentEthicalGatheringGame/commits" | grep -E -m 1 '"sha"|"date"' | sed -E 's/.*: "(.*)".*/\1/'
echo "===================="

srun python main.py --tag epochs_findings --seed 3 --n-steps 2000 --n-epochs 20