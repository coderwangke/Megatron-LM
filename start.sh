#! /bin/bash

####################################################
#
# usage:
#      bash start.sh <master_addr> <node_num> <rank> <model_size>
#
# supported model size: {7, 30, 60}
#
####################################################

# env var
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# nccl settings
for i in `ibdev2netdev | awk '{print $1}'`
do
        hca_list+="$i,"
done
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=${hca_list::-1}
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0

export GLOO_SOCKET_IFNAME=eth0

# node settings
MASTER_ADDR=${1:-localhost}
MASTER_PORT=6000
NNODES=${2:-1}
NODE_RANK=${3:-0}
GPUS_PER_NODE=8
WORLD_SIZE=$(( $GPUS_PER_NODE * $NNODES ))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# data settings
# get the value of the environment variable BASE_DATA_PATH
BASE_DATA_PATH="$BASE_DATA_PATH"
DATA_PATH=$BASE_DATA_PATH/my-gpt2_text_document
VOCAB_FILE=$BASE_DATA_PATH/gpt2-vocab.json
MERGE_FILE=$BASE_DATA_PATH/gpt2-merges.txt
CHECKPOINT_PATH=$DIR/checkpoint

# model settings
SEQ_LEN=2048
MAX_SEQ_LEN=2048
MODEL_SIZE=${4:-7}
if [ $MODEL_SIZE == "7" ]; then
        NUM_LAYERS=32
        HIDDEN_SIZE=4096
        NUM_ATTN_HEADS=32
        MICRO_BATCH_SIZE=20
	TP=4
	PP=2
	MICRO_BATCH_NUM=16
elif [ $MODEL_SIZE == "30" ]; then
        NUM_LAYERS=48
        HIDDEN_SIZE=7168
        NUM_ATTN_HEADS=56
	MICRO_BATCH_SIZE=4
	TP=4
	PP=2
	MICRO_BATCH_NUM=16
elif [ $MODEL_SIZE == "60" ]; then
        NUM_LAYERS=80
        HIDDEN_SIZE=8192
        NUM_ATTN_HEADS=64
	MICRO_BATCH_SIZE=4
	TP=8
	PP=4
	MICRO_BATCH_NUM=32
else
        echo "ERROR: Please supplement new model configuration to test!"
        exit -1
fi

DP=$(( $WORLD_SIZE / $TP / $PP ))
GLOBAL_BATCH_SIZE=$(( $DP * $MICRO_BATCH_SIZE * $MICRO_BATCH_NUM ))

#fp8 settings
ENABLE_FP8=false
if [ $ENABLE_FP8 == "true" ]; then
	FP8_OPTS="--transformer-impl transformer_engine \
       	          --fp8-hybrid \
                  --fp8-amax-history-len 32 \
                  --fp8-amax-compute-algo max"
        DT="fp8"
else
	FP8_OPTS=""
        DT="bf16"
fi

CMD="torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --sequence-parallel \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTN_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --exit-interval 10000 \
       --eval-iters 1000 \
       --use-flash-attn \
       --recompute-activations \
       --use-distributed-optimizer \
       --bf16 \
       $FP8_OPTS \
       "

echo ${CMD} 2>&1 | tee megatron_gpt${MODEL_SIZE}B_tp${TP}_pp${PP}_dp${DP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${DT}.log
eval ${CMD} 2>&1 | tee megatron_gpt${MODEL_SIZE}B_tp${TP}_pp${PP}_dp${DP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${DT}.log