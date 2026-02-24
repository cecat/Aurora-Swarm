#!/bin/bash
#
if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

python ../scripts/wait_for_vllm_servers.py --hostfile $HOSTFILE --output ../scripts/swarm.hostfile

# batch size is 2000 prompts sent across assuming 4 servers,
# then each vllm requests queue will get batches of 500 prompts.
python ./scatter_gather_coli.py /home/brettin/ModCon/brettin/Aurora-Inferencing/examples/TOM.COLI/batch_1/ --hostfile ../scripts/swarm.hostfile --num-files 16 --output test.openai.batch.txt --model openai/gpt-oss-120b --batch-size 2000
