#!/usr/bin/bash
. ./common.sh

#abs_path=`dirname $(readlink -f $0)`			# 执行脚本的绝对路径
#echo $abs_path
#py_file="$(dirname $(dirname $abs_path))/main.py"

sh_name=`echo $0 | awk -F'.sh' '{print $1}'`
YESTERDAY=`date -d "1 days ago" '+%Y%m%d'`

train_path="dssm_data/train"
eval_path="dssm_data/eval"
test_path="dssm_data/test"
vec_embedding="dssm_data/vec_embedding"
image_embedding="dssm_data/image_embedding"
title_embedding="dssm_data/title_embedding"

py_file="../../main.py"
schema_file="conf/dssm_schema.csv"
model_dir="dssm_model/"
log_file="logs/${sh_name}_train.log_$YESTERDAY"

get_data(){
    download_dir  "obs://sprs-data-sg/sharezone/chengtx/dssm_data/train" $train_path
    download_dir  "obs://sprs-data-sg/sharezone/chengtx/dssm_data/eval"  $eval_path
    download_dir  "obs://sprs-data-sg/sharezone/chengtx/dssm_data/test"  $test_path

	#download_file "obs://sprs-data-sg/sharezone/chengtx/dssm_data/vec_embedding"    $vec_embedding
	#download_file "obs://sprs-data-sg/sharezone/chengtx/dssm_data/image_embedding"  $image_embedding
	#download_file "obs://sprs-data-sg/sharezone/chengtx/dssm_data/title_embedding"  $title_embedding
}

train_model(){
    CUDA_DEVICE_ID="9"
	rm -rf $model_dir && mkdir $model_dir
    cmd="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${py_file} 
        --model two_tower
        --dataset two_tower
        --network dnn
        --hidden 256,128,64
		--vector_dim 32
		--checkpoints_steps 10000
        --schema $schema_file
        --compression_type GZIP
        --train_path $train_path
        --valid_path $eval_path
        --mode train
        --learning_rate 0.01 
        --epochs 8
        --prebatch 1
        --batch_size 512 
        --checkpoint_dir ${model_dir}
        --fixed_emb_dim 16
        --label_key played_duration
		--label_dtype float
		--label_num 6
		"
        #> ${log_file} 2>&1"
    echo $cmd
	eval $cmd
}

echo "========== BEGIN `date` =========="
get_data
train_model
echo "========== FINISH `date` =========="
