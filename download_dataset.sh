#!/bin/bash

# Get the value of the environment variable BASE_DATA_PATH
DATA_BASEPATH="$BASE_DATA_PATH"

MERGES_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/gpt_data/gpt2-merges.txt"
VOCAB_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/gpt_data/gpt2-vocab.json"
DOC_BIN_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/gpt_data/my-gpt2_text_document.bin"
DOC_IDX_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/gpt_data/my-gpt2_text_document.idx"

if [ -f "$DATA_BASEPATH/gpt2-merges.txt" ]; then
	echo "The dataset has been already downloaded for training."
	exit 0
else
	if [ ! -d "$DATA_BASEPATH" ]; then
		mkdir -p data
	fi
	wget $MERGES_URL -P $DATA_BASEPATH -q --show-progress
	wget $VOCAB_URL -P $DATA_BASEPATH -q --show-progress
	wget $DOC_BIN_URL -P $DATA_BASEPATH -q --show-progress
	wget $DOC_IDX_URL -P $DATA_BASEPATH -q --show-progress
fi