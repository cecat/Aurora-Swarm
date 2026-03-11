#!/usr/bin/env bash
# Launch the tree-reduce COLI example.
# Usage: ./tree_reduce_coli.sh <hostfile>
# Run from the examples/ directory. Waits for vLLM servers, then runs
# tree-reduce over bacterial genes (leaf: scatter_gather per chunk file;
# reduce: groups of gene responses by --fanin, hierarchical synthesis).

if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

# Run from examples/; env.sh is at project root
source ../env.sh

# Token limits and reduce group size (override via env)
MAX_TOKENS=${MAX_TOKENS:-2024} # Leaf max tokens
MAX_TOKENS_CONTINGENCY=${MAX_TOKENS_CONTINGENCY:-4096}
FANIN=${FANIN:-12}

# Design.
# We want fanin to be 12 for now.
# I think max tokens is the is the input and output tokens.
# So we need to make sure that the input and output tokens are less than the max tokens.
# For the leaf output, we make that 2024 tokens.
# This means 16*2024 = 32384 tokens the fanin can handle.



python ../scripts/wait_for_vllm_servers.py --hostfile "$HOSTFILE" --output ../scripts/swarm.hostfile

python ./tree_reduce_coli.py /home/brettin/ModCon/brettin/Aurora-Inferencing/examples/TOM.COLI/batch_1/ \
    --hostfile ../scripts/swarm.hostfile \
    --num-files 16 \
    --fanin "$FANIN" \
    --max-tokens "$MAX_TOKENS" \
    --max-tokens-contingency "$MAX_TOKENS_CONTINGENCY" \
    --output tree_reduce_coli_output.txt \
    --model openai/gpt-oss-120b \
    --monitor-sockets \
    --socket-interval 10 \
    --plot-sockets \
    --plot-output socket_usage.png
