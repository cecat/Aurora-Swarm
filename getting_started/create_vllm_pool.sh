

if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

source ../env.sh > /dev/null 2>&1

python ../scripts/wait_for_vllm_servers.py \
	--hostfile "$HOSTFILE" \
	--output ../scripts/swarm.hostfile

python ./create_vllm_pool.py ../scripts/swarm.hostfile

python ./create_vllm_subpools.py ../scripts/swarm.hostfile
