#!/bin/bash
#PBS -N tree_reduce 
#PBS -l walltime=01:00:00
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -o 128.output.log
#PBS -e 129.error.log
#PBS -l select=128
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -j oe

# Input/Output configuration
SCRIPT_DIR="/lus/flare/projects/ModCon/brettin/Aurora-Swarm/scripts"
MODEL_PATH="/lus/flare/projects/datasets/model-weights/hub/models--openai--gpt-oss-120b"
CONDA_ENV_PATH="/lus/flare/projects/ModCon/brettin/Aurora-Inferencing/vllm-gpt-oss120b/vllm_oss_conda_pack_01082026.tar.gz"

# Extract model name from MODEL_PATH (converts models--org--name to org/name)
MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/^models--//' | sed 's/--/\//')

# Operation settings
STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}     # 1=stage model weights to /tmp, 0=skip staging
STAGE_CONDA=${STAGE_CONDA:-1}         # 1=stage conda environment to /tmp, 0=skip staging
USE_FRAMEWORKS=${USE_FRAMEWORKS:-0}   # 1=use frameworks module, 0=use conda environment
BLACKBOARD=${BLACKBOARD:-1}           # 1=use blackboard mode, 0=use normal mode

# vLLM server settings
VLLM_HOST_PORT=${VLLM_HOST_PORT:-6739}

# SSH and timing settings
SSH_TIMEOUT=10                        # SSH connection timeout in seconds

# Functions
start_vllm_on_host() {
    local host=$1
    local model=$2
    local port=$3
    if ! ssh -o ConnectTimeout="${SSH_TIMEOUT}" -o StrictHostKeyChecking=no "$host" "bash -c 'cd $SCRIPT_DIR && export USE_FRAMEWORKS=${USE_FRAMEWORKS} && export VLLM_HOST_PORT=${port} && ./start_oss120b.sh \"$model\"'" 2>&1; then
        echo "$(date) Failed to launch vLLM on $host (model: $model)"
        return 1
    fi
}

# Main Execution


# Write host and port in tab-delimited format to hostfile.
# Truncates (empties) the hostfile to prepare it for writing a fresh list of hosts for this run.
if [ "$BLACKBOARD" -eq 1 ]; then
    # Read nodes into an array and remove domain name
    mapfile -t all_nodes < <(cut -d'.' -f1 "$PBS_NODEFILE")
    total_nodes=${#all_nodes[@]}
    half_nodes=$((total_nodes / 2))
    # If odd, first half will be 1 greater
    if [ $((total_nodes % 2)) -ne 0 ]; then
        half_nodes=$((half_nodes + 1))
    fi
    : > "$SCRIPT_DIR/hostfile"  # Truncate before writing

    for i in "${!all_nodes[@]}"; do
        node="${all_nodes[$i]}"
        if [ "$i" -lt "$half_nodes" ]; then
            # First half - role=hypotheses
            echo -e "${node}\t${VLLM_HOST_PORT}\trole=hypotheses" >> "$SCRIPT_DIR/hostfile"
        else
            # Second half - role=critiques
            echo -e "${node}\t${VLLM_HOST_PORT}\trole=critiques" >> "$SCRIPT_DIR/hostfile"
        fi
    done
else
    : > "$SCRIPT_DIR/hostfile"
    while read -r node; do
        echo -e "${node}\t${VLLM_HOST_PORT}" >> "$SCRIPT_DIR/hostfile"
    done < "$PBS_NODEFILE"
fi

echo "$(date) vLLM Multi-Node Deployment"
echo "$(date) Script directory: $SCRIPT_DIR"
echo "$(date) PBS Job ID: $PBS_JOBID"
echo "$(date) PBS Job Name: $PBS_JOBNAME"
echo "$(date) Nodes allocated: $(wc -l < $PBS_NODEFILE)"
echo "$(date) Model: $MODEL_NAME"
echo "$(date) VLLM_HOST_PORT: $VLLM_HOST_PORT"
echo "$(date) Model path: $MODEL_PATH"
echo "$(date) STAGE_WEIGHTS: $STAGE_WEIGHTS"
echo "$(date) STAGE_CONDA: $STAGE_CONDA"
echo "$(date) CONDA_ENV_PATH = ${CONDA_ENV_PATH}"
echo "$(date) USE_FRAMEWORKS: $USE_FRAMEWORKS"

# Create array containing hostnames (without domain suffix)
mapfile -t hosts < <(cut -d'.' -f1 "$PBS_NODEFILE")
total_hosts=${#hosts[@]}

echo "$(date) Hosts: ${hosts[@]}"

# Stage Model Weights
if [ "$STAGE_WEIGHTS" -eq 1 ]; then
    echo "$(date) Staging model weights to /tmp on all nodes"
    mpicc -o "${SCRIPT_DIR}/../cptotmp" "${SCRIPT_DIR}/../cptotmp.c"
    export MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING=1
    export MPIR_CVAR_CH4_OFI_MAX_NICS=4
    time mpiexec -ppn 1 --cpu-bind numa "${SCRIPT_DIR}/../cptotmp" "$MODEL_PATH" /tmp/hf_home/hub/ 2>&1 || \
        echo "$(date) WARNING: Model staging failed or directory not found, will use shared filesystem"
    echo "$(date) Model staging complete"
fi

# Stage Conda Environment - right now, it uses the cptotmp default location of /tmp/hf_home/hub
if [ "$STAGE_CONDA" -eq 1 ]; then
    echo "$(date) Staging conda environment to /tmp on all nodes"
    if [ ! -f "${SCRIPT_DIR}/../cptotmp" ]; then
        mpicc -o "${SCRIPT_DIR}/../cptotmp" "${SCRIPT_DIR}/../cptotmp.c"
    fi
    export MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING=1
    export MPIR_CVAR_CH4_OFI_MAX_NICS=4
    time mpiexec -ppn 1 --cpu-bind numa "${SCRIPT_DIR}/../cptotmp" "$CONDA_ENV_PATH" 2>&1 || \
        echo "$(date) WARNING: Conda environment staging failed or directory not found, will use shared filesystem"
    echo "$(date) Conda environment staging complete"

    # Unpack Conda Environment in parallel on all nodes
    echo "$(date) Unpacking conda environment on all nodes in parallel"
    time mpiexec -ppn 1 --cpu-bind numa bash -c 'mkdir -p /tmp/hf_home/hub/vllm_env && tar -xzf /tmp/vllm_oss_conda_pack_01082026.tar.gz -C /tmp/hf_home/hub/vllm_env' 2>&1 || \ 
        echo "$(date) WARNING: Conda environment unpacking failed"
    echo "$(date) Conda environment unpacking complete"
fi


# Launch vLLM on Each Host
declare -a pids
declare -a launch_hosts

for ((i = 0; i < total_hosts; i++)); do
    host="${hosts[$i]}"

    # Launch vLLM on this host
    start_vllm_on_host "$host" "$MODEL_NAME" "$VLLM_HOST_PORT" &
    pid=$!
    pids+=($pid)
    launch_hosts+=("$host")

done

# Wait for Completion
echo "$(date) All launches initiated, waiting for completion..."
success_count=0
failed_count=0

for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    host=${launch_hosts[$i]}

    echo "$(date) Waiting for $host (PID: $pid)"
    if wait $pid; then
        echo "$(date) $host completed successfully"
        ((success_count++))
    else
        echo "$(date) $host FAILED with exit code $?"
        ((failed_count++))
    fi
done

echo "$(date) Deployment Summary"
echo "$(date) Total nodes:      $total_hosts"
echo "$(date) Successful:       $success_count"
echo "$(date) Failed:           $failed_count"
echo "$(date)"

exit 0
