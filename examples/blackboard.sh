#!/usr/bin/env bash
# Launch the broadcast + aggregators example.
# Usage: ./broadcast_aggregators.sh <hostfile>
# Run from the examples/ directory. Waits for vLLM servers, then broadcasts
# to all hosts except one and runs majority_vote and concat aggregators.

if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

python ../scripts/wait_for_vllm_servers.py --hostfile "$HOSTFILE" --output ../scripts/swarm.hostfile

python ./blackboard_example.py --hostfile ../scripts/swarm.hostfile --model openai/gpt-oss-120b --log-dir ./logs

