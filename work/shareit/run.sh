#!/usr/bin/bash
# 启动脚本
. ../commont_utils/common.sh
export TZ='Asia/Shanghai'
main_py="../../main.py"
embed_py="../commont_utils/export_embedding.py"

get_user_emb_data()
{
   dt=$1
   log_path=$2
   user_embeding_path="${OBS_USER_EMB_PATH}/${dt}/eval_data"
   local_user_embeding_path="${LOCAL_DATA_DIR}/${dt}"
   rm -rf ${local_user_embeding_path}/$dt/user
   download_dir "${user_embeding_path}" $local_user_embeding_path/
   mv ${local_user_embeding_path}/eval_data ${local_user_embeding_path}/user
}



# 获取训练数据
get_train_data(){
    dt=$1
    log_path=$2
    echo -e "----- $FUNCNAME BEGIN `date` -----"

    data_dir="${LOCAL_DATA_DIR}/${dt}"
    train_dir="${data_dir}/train/"
    eval_dir="${data_dir}/eval/"
    test_dir="${data_dir}/test/"
    rm -rf $train_dir $eval_dir && mkdir -p $train_dir && mkdir -p $eval_dir

    log_info "train_dir = ${train_dir}" ${log_path}
    download_dir "${OBS_DATA_DIR}/${dt}/train_data" $train_dir
    log_info "eval_dir = ${eval_dir}" ${log_path}
    download_dir "${OBS_DATA_DIR}/${dt}/pred_data" $eval_dir
    if [ ! -z ${OBS_PRED_DATA_DIR} ]; then
        log_info "test_dir = ${test_dir}" ${log_path}
		download_dir "${OBS_PRED_DATA_DIR}/${dt}/test_data" ${test_dir}
	fi

    data_size=`du -s $data_dir | awk '{print $1}'`
    if [ $data_size -lt 10240 ]; then
        log_error "$data_dir is only $data_size bytes, please check data" ${log_path}
        return 1
    fi
    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}

clean_data(){
	dt=$1
	log_path=$2

	echo -e "----- $FUNCNAME BEGIN `date` -----"

	data_dir="${LOCAL_DATA_DIR}/${dt}/"
	model_dir="${LOCAL_MODEL_DIR}/${dt}"
	model_name=`echo ${LOCAL_MODEL_DIR} | awk -F'/' '{print $NF}'`
	log_info "old_data_dir = ${data_dir}" ${log_path}
	log_info "old_model_dir = ${model_dir}" ${log_path}
	rm -rf $data_dir $model_dir 
	echo -e "----- $FUNCNAME FINISH `date` -----\n"
}


function monitor_test_after_push_model()
{
    dt=$1
    log_path=$2
    
    valid_flag=${TST_VALID_FLAG}
    key_type="test"

    model_name=`echo ${LOCAL_MODEL_DIR} | awk -F'/' '{print $NF}'`          # 根据model_dir获取训练模型的名称

    save_path="${LOCAL_MODEL_DIR}/${dt}/monitor_${key_type}.txt"
    rm -f ${save_path}
    
    dict_txt="${EXPORT_MODEL_DIR}/${ONLINE_DICT_NAME}.txt"
    feature_num=`wc -l ${dict_txt} | awk '{print $1}'`
    model_size=`du -b ${EXPORT_MODEL_DIR}/variables | awk '{print $1}'`

    echo "[monitor] model_size=${model_size}"   >> ${save_path}
    echo "[monitor] feature_num=${feature_num}" >> ${save_path}

    log_info=`grep "${valid_flag}" ${log_path} | tail -1 | awk '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`

    for item in ${log_info}
    do
        echo "[monitor] ${item}" >> ${save_path}
    done
    
    base_key="SPRS.Model.daily.${ONLINE_DICT_NAME}.${key_type}"

    log_info "Starting importing monitor of ${dt}" ${log_path}
    cat ${save_path} | python3 ${RUN_DIR}/scripts/import.py "${base_key}" "${dt}" >> ${log_path} 2>&1
    log_info "Finished importing monitor of ${dt}, ret=${is_model_valid}" ${log_path}
}

# 训练模型
train_model(){
    dt=$1
    mode=$2
    log_path=$3

    ori_mode=${mode}
    echo -e "----- $FUNCNAME BEGIN `date` -----"

    train_dir="${LOCAL_DATA_DIR}/${dt}/train/"
    eval_dir="${LOCAL_DATA_DIR}/${dt}/eval/"
    user_dir="${LOCAL_DATA_DIR}/${dt}/user/"
    test_dir="${LOCAL_DATA_DIR}/${dt}/test/"
    pred_path="${LOCAL_MODEL_DIR}/${dt}/pred.txt"
    model_dir="${LOCAL_MODEL_DIR}/${dt}"
    model_name=`echo ${LOCAL_MODEL_DIR} | awk -F'/' '{print $NF}'`          # 根据model_dir获取训练模型的名称
    prebatch_num=${PREBATCH_NUM}
    batch_size=${BATCH_SIZE}
    debug_labels=${DEBUG_LABELS}
    pred_with_emb=""

    case $mode in
        "train")
             rm -rf $model_dir && mkdir -p $model_dir
             ;;
        "test")
             ;;
        "predict")
             dt_yes=`date -d "$dt -1 days" +%Y%m%d`
             model_dir="${LOCAL_MODEL_DIR}/${dt_yes}"
             ;;
        "user_embedding")
            mode="predict"
            eval_dir="${user_dir}"
            prebatch_num=512
            batch_size=1024
            pred_path="${LOCAL_MODEL_DIR}/${dt}/embedding_${EXPORT_VARIABLE_USER}"
            pred_with_emb="--predict_with_user"
            debug_labels=${DEBUG_LABELS_USER}
            ;;
        "export_new")
            mode="predict"
            eval_dir="${test_dir}"
            prebatch_num=1
            batch_size=1024
            pred_path="${LOCAL_MODEL_DIR}/${dt}/embedding_${EXPORT_VARIABLE}"
            pred_with_emb="--predict_with_emb"
            debug_labels=${DEBUG_LABELS_EMB}
            ;;
        *)
             log_error "$mode not support!" ${log_path}
             return 1
             ;;
    esac

    cmd="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${main_py} 
        --model ${MODEL}
        --dataset ${DATASET} 
        --network ${NETWORK}
        --hidden ${HIDDEN_UNITS}
        --activation ${ACTIVATION}
        --vector_dim ${OUTLAYER_UNIT}
        --schema ${LOCAL_MODEL_SCHEMA}
        --compression_type GZIP
        --train_path ${train_dir}
        --valid_path ${eval_dir}
        --pred_path ${pred_path}
        --mode ${mode}
        --learning_rate ${LEARNING_RATE}
        --epochs ${EPOCH_NUM}
        --prebatch ${prebatch_num}
        --batch_size ${batch_size}
        --checkpoint_dir ${model_dir}
        --fixed_emb_dim ${FIXED_EMBEDDING_DIM}
        --checkpoints_steps ${CHECKPOINT_STEPS}
        --fixed_training_steps ${FIXED_TRAINING_STEPS}
        --warm_start_mode all 
        --warm_start_dir ${WARM_START_DIR}
        --label_key ${LABEL_KEY}
        --label_dtype ${LABEL_TYPE}
        --label_num ${LABEL_NUM}
        --debug_labels ${debug_labels}
        --negative_rate ${NEGATIVE_RATE}
        ${pred_with_emb}
        >> ${log_path} 2>&1
        "
    log_info "train_dir = ${train_dir}" ${log_path}
    log_info "eval_dir  = ${eval_dir}" ${log_path}
    log_info "model_dir = ${model_dir}" ${log_path}
    log_info "train_cmd = ${cmd}" ${log_path}

    eval $cmd
    if [ $? -ne 0 ]; then
        return 1
    fi

    if [ "${ori_mode}" == "predict" ]; then
        cat ${pred_path} | python3 ${RUN_DIR}/scripts/auc.py main ${PRED_OUTPUT_FORMAT} ${TST_VALID_FLAG} >> ${log_path} 2>&1
    fi

    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}


# 模型备份，用于增量训练
backup_checkpoint(){
    dt=$1
    log_path=$2

    echo -e "----- $FUNCNAME BEGIN `date` -----"

    model_dir="${LOCAL_MODEL_DIR}/${dt}"
    if [ ! -f "${model_dir}/checkpoint" ]; then
        log_error "${model_dir}/checkpoint not exist!" ${log_path}
        return 1
    fi
    if [ ! -f "${model_dir}/graph.pbtxt" ]; then
        log_error "${model_dir}/graph.pbtxt not exist!" ${log_path}
        return 1
    fi

    rm -rf ${INCREMENTAL_MODEL_DIR} && mkdir -p ${INCREMENTAL_MODEL_DIR}
    cp -r ${model_dir}/checkpoint  ${INCREMENTAL_MODEL_DIR}/
    cp -r ${model_dir}/graph.pbtxt ${INCREMENTAL_MODEL_DIR}/
    cp -r ${model_dir}/model.*     ${INCREMENTAL_MODEL_DIR}/

    log_info "backup checkpoint from ${model_dir} to ${INCREMENTAL_MODEL_DIR}" ${log_path}
    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}


# 导出指定变量的embedding
export_embedding(){
  dt=$1
  log_path=$2

  echo -e "----- $FUNCNAME BEGIN `date` -----"
  export_source=${EXPORT_SOURCE}
  out_file="${LOCAL_MODEL_DIR}/${dt}/embedding_${EXPORT_VARIABLE}"
  case $export_source in
    "model"):
      model_ts=`ls -tr ${LOCAL_MODEL_DIR}/${dt}/export/best | awk '{print $1}'`
      best_model_dir="${LOCAL_MODEL_DIR}/${dt}/export/best/${model_ts}"
      cmd="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${embed_py} --model_dir $best_model_dir --variable ${EXPORT_VARIABLE} --out_file $out_file"
      ;;
      "checkpoint"):
      checkpoint_dir="${LOCAL_MODEL_DIR}/${dt}"
      cmd="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${embed_py} --checkpoint_dir $checkpoint_dir --variable ${EXPORT_VARIABLE} --out_file $out_file"
      ;;
      *):
      log_error "$export_source not support!" ${log_path}
      return 1
      ;;
  esac

  log_info "export_cmd = $cmd"
  eval $cmd
  log_info "exported ${EXPORT_VARIABLE} from $export_source to $out_file" ${log_path}

  echo -e "----- $FUNCNAME FINISH `date` -----\n"
}



# 导出指定变量的embedding
save_item_embedding(){
    dt=$1
    log_path=$2

    echo -e "----- $FUNCNAME BEGIN `date` -----"
    item_index_file="${LOCAL_DATA_DIR}/${dt}/item_index"
    log_info "item_index_file = $item_index_file" ${log_path}
    download_file "${OBS_DATA_DIR}/${dt}/item_index" $item_index_file
    if [ -f $item_index_file ]; then
        file_num=`wc -l $item_index_file | awk '{print $1}'`
        if [ $file_num -lt 100000 ]; then
            log_error "$item_index_file only has $file_num items!" ${log_path}
            return 1
        fi
    else
        log_error "get $item_index_file fail!" ${log_path}
        return 1
    fi

    # 历史的item embedding
    dt_yes=`date -d "$dt -1 days" +%Y%m%d`
    old_item_embedding=""
    if [ -d "${LOCAL_MODEL_DIR}/${dt_yes}/export/best" ]; then
        old_model_ts=`ls -tr ${LOCAL_MODEL_DIR}/${dt_yes}/export/best | awk '{print $1}'`
        old_item_embedding="${LOCAL_MODEL_DIR}/${dt_yes}/export/best/${old_model_ts}/item_embedding"
    fi

    # index embedding文件
    index_embedding_file="${LOCAL_MODEL_DIR}/${dt}/embedding_${EXPORT_VARIABLE}"
    if [ ! -f $index_embedding_file ]; then
        export_embedding $dt
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi

    # 生成最新的item embedding
    model_ts=`ls -tr ${LOCAL_MODEL_DIR}/${dt}/export/best | awk '{print $1}'`
    best_model_dir="${LOCAL_MODEL_DIR}/${dt}/export/best/${model_ts}"
    item_embedding_file="${best_model_dir}/item_embedding"
    cmd="python item_embedding.py $item_index_file $index_embedding_file $item_embedding_file"
    if [[ x"" != x"$old_item_embedding" ]] && [[ -f $old_item_embedding ]]; then
        cmd="$cmd --old_embedding $old_item_embedding"
    fi
	log_info "save_cmd = $cmd" ${log_path}
    eval $cmd
    log_info "saved item embedding to $item_embedding_file" ${log_path}

    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}


# 导出模型用于线上servering
push_model(){
    dt=$1
    log_path=$2

    echo -e "----- $FUNCNAME BEGIN `date` -----"
    model_ts=`ls -tr ${LOCAL_MODEL_DIR}/${dt}/export/best | awk '{print $1}'`
    if [ $? -ne 0 ]; then
        log_error "${LOCAL_MODEL_DIR}/${dt}/export/best not exist!" ${log_path}
        return 1
    fi
    best_model_dir="${LOCAL_MODEL_DIR}/${dt}/export/best/${model_ts}"
    if [ ! -f "${best_model_dir}/saved_model.pb" ]; then
        log_error "${best_model_dir}/saved_model.pb not exist!" ${log_path}
        return 1
    fi  
    if [ ! -d "${best_model_dir}/assets.extra" ]; then
        log_error "${best_model_dir}/assets.extra not exist!" ${log_path}
        return 1
    fi  
    if [ ! -d "${best_model_dir}/variables" ]; then
        log_error "${best_model_dir}/variables not exist!" ${log_path}
        return 1
    fi  

    rm -rf ${EXPORT_MODEL_DIR} && mkdir -p ${EXPORT_MODEL_DIR}
    log_info "export_model_dir = ${EXPORT_MODEL_DIR}" ${log_path}
    # 复制模型
    cp -r ${best_model_dir}/* ${EXPORT_MODEL_DIR}
    # 生成schema文件以及md5文件
    schema_txt="${EXPORT_MODEL_DIR}/${ONLINE_DICT_NAME}.txt"
    schema_md5="${EXPORT_MODEL_DIR}/${ONLINE_DICT_NAME}.md5"
    if [ "${MODEL}" == "dssm" ]; then
      cat ${LOCAL_MODEL_SCHEMA} | awk -F ',' '{if ($6==0) print $1"\t"tolower($2)"\t"$3"\t"$4"\t"$5}' > $schema_txt
    else
      cat ${LOCAL_MODEL_SCHEMA} | awk -F ',' '{print $1"\t"tolower($2)"\t"$3"\t"$4"\t"$5}' > $schema_txt
    fi
    echo -e "dnn_version\t${dt}" >> ${schema_txt}
    md5sum ${schema_txt} | awk '{print $1}' >> ${schema_md5}

    # 推送到线上
	  online_model_dir="${ONLINE_MODEL_DIR}/${ONLINE_DICT_NAME}/${dt}/"
	  log_info "online_model_dir = $online_model_dir" ${log_path}
    upload_file ${EXPORT_MODEL_DIR} $online_model_dir
    if [ $? -ne 0 ]; then
        log_error "upload ${EXPORT_MODEL_DIR} to $online_model_dir fail!"
        delete_file $online_dir
        return 1
    fi
    log_info "Finished pushing model to HuaweiCloud" ${log_path}

    rm_date=`date -d "${version} -3days" +"%Y%m%d"`
    if [[ ${ONLINE_MODEL_DIR} =~ /$ ]]; then
        rm_online_model="${ONLINE_MODEL_DIR}${ONLINE_DICT_NAME}/${rm_date}"
    else
        rm_online_model="${ONLINE_MODEL_DIR}/${ONLINE_DICT_NAME}/${rm_date}"
    fi
    /root/obsutil/obsutil rm -f -r ${rm_online_model} >>${log_path}

    log_info "Delete online_model=${rm_online_model}" ${log_path}
    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}

# 导出多语言的item_embedding
save_item_embedding_new(){
    dt=$1
    log_path=$2

    echo -e "----- $FUNCNAME BEGIN `date` -----"
    # 下载item_index文件
    # index embedding文件
    embedding_file="${LOCAL_MODEL_DIR}/${dt}/embedding_${EXPORT_VARIABLE}"
    
	# 生成最新的item embedding
    model_ts=`ls -tr ${LOCAL_MODEL_DIR}/${dt}/export/best | awk '{print $1}'`
    best_model_dir="${LOCAL_MODEL_DIR}/${dt}/export/best/${model_ts}"
    item_embedding_file="${best_model_dir}"
    cmd="python item_lang_embedding.py $embedding_file $item_embedding_file $DEBUG_LABELS_EMB"
    log_info "save_cmd = $cmd" ${log_path}
    eval $cmd
    log_info "saved item embedding to $item_embedding_file" ${log_path}

    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}

# 验证特征重要性
function eval_feature(){
    dt=$1
    log_path=$2
    feature_name=$3

    if [ "$feature_name" == '' ]; then
        feature_name='None'
        log_info "Starting eval with all feature" ${log_path}
    else
        log_info "Starting eval without feature $feature_name" ${log_path}
    fi
    echo -e "----- $FUNCNAME BEGIN `date` -----"

    train_dir="${LOCAL_DATA_DIR}/${dt}/train/"
    eval_dir="${LOCAL_DATA_DIR}/${dt}/eval/"
    model_dir="${LOCAL_MODEL_DIR}/${dt}"

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${main_py} \
        --model ${MODEL} \
        --feature_influence $feature_name \
        --dataset ${DATASET} \
        --network ${NETWORK} \
        --hidden ${HIDDEN_UNITS} \
        --activation ${ACTIVATION} \
        --vector_dim ${OUTLAYER_UNIT} \
        --schema ${LOCAL_MODEL_SCHEMA} \
        --compression_type GZIP \
        --train_path ${train_dir} \
        --valid_path ${eval_dir} \
        --mode test \
        --learning_rate ${LEARNING_RATE} \
        --epochs ${EPOCH_NUM} \
        --prebatch ${PREBATCH_NUM} \
        --batch_size ${BATCH_SIZE} \
        --checkpoint_dir ${model_dir} \
        --fixed_emb_dim ${FIXED_EMBEDDING_DIM} \
        --checkpoints_steps ${CHECKPOINT_STEPS} \
        --fixed_training_steps ${FIXED_TRAINING_STEPS} \
        --warm_start_mode all \
        --warm_start_dir ${WARM_START_DIR} \
        --label_key ${LABEL_KEY} \
        --label_dtype ${LABEL_TYPE} \
        --label_num ${LABEL_NUM} \
        --debug_labels ${DEBUG_LABELS} \
        --negative_rate ${NEGATIVE_RATE} \
        > ${log_path} 2>&1

    echo -e "----- $FUNCNAME FINISH `date` -----\n"
}


# 提取特征重要性具体指标
function _influence_out() {
  influence_file=$1
  feature_name=$2
  log_path=$3

  score_info=`grep "Saving dict for global step" ${log_path} | tail -n 1 | sed s/[[:space:]]//g | awk -F ':|,' '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`
  for item in ${score_info}
  do
      key=`echo "${item}" | awk -F '=' '{print $1}'`
      val=`echo "${item}" | awk -F '=' '{print $2}'`
      if [ "${key}" == "roc_auc" ]; then
          auc=${val}
      fi
      if [ "${key}" == "loss" ]; then
          loss=${val}
      fi
      if [ "${key}" == "score" ]; then
          score=${val}
      fi
  done
  echo "$feature_name,$auc,$loss,$score" >> $influence_file
}


# 计算特征重要性，并将结果通过邮件发送
function feature_influence()
{
    dt=$1
    receviers=$2
    log_path=$3

    model_name=`echo ${LOCAL_MODEL_DIR} | awk -F'/' '{print $NF}'`          # 根据model_dir获取训练模型的名称
    influence="${LOCAL_DATA_DIR}/${dt}/${dt}_${model_name}_feature_influence.csv"
    log_info "$influence" ${log_path}
    rm -f $influence
    echo "name,auc,loss,score" > $influence
    log_info "Starting feature influence" ${log_path}
    eval_feature ${dt} ${log_path}
    _influence_out $influence "base" ${log_path}

    features=`cat ${LOCAL_MODEL_SCHEMA} | awk -F ',' '{print $1}'`
    for feat in ${features}
    do
        eval_feature ${dt} ${log_path} $feat
        _influence_out $influence $feat ${log_path}
    done
    python3 ./calculate_influence.py $influence
     if [ "$receviers" != '' ]; then
        feature_name='None'
        python3 ../commont_utils/send_mail.py $receviers "[TF_TRAINER]_${dt}_${model_name}_RECALL_feat_influence" "" $influence
        log_info "Send influence.csv success" ${log_path}
    fi
    log_info "Done feature influence" ${log_path}
}

Usage(){
 cat <<EOF

Usage:
    $0 [conf_file] [task_type] [task_date]

    conf_file is required
    task_type is required
    task_date is optional, default yesterday

  task_type:
    download        Download train and eval data from obs to local. Local data dir and obs data path is specified in conf_file.
    train           Train model. Hyper-parameter is specified in conf_file.
    eval            Evaluate model.
    predict         Predict model.
    backup          Backup checkpoint. The backup directory is specified in conf_file
    export          Export specified variable embeddings.
    export_new      Export specified variable embeddings using predict method.
    push_model      Export then best model and push it to online server.
    clean_data      Clean expired data.
    feat_influence  Eval feature influence
    save_item_embedding_new     Get item embeddings in multiple languages
    monitor         Check model and upload monitor indicators

    
EOF
    exit 1
}

export TRN_VALID_FLAG="Final-train-valid"
export TST_VALID_FLAG="Final-test-valid"

[ $# -lt 2 ] && Usage

conf_file=$1
task_type=$2
task_date=`date -d "1 days ago" +%Y%m%d`
if [ $# -ge 3 ]; then
    task_date=$3
fi
receviers=""
if [ $# -ge 4 ]; then
    receviers=$4
fi

source ${conf_file}

work_dir="${LOCAL_MODEL_DIR}/${task_date}"
if [ ! -d ${work_dir} ]; then
    mkdir -p ${work_dir}
fi
log_path="${work_dir}/run.log"

if [ ! -f ${conf_file} ]; then
    echo "${conf_file} not exist!"
    exit 0
fi

case $task_type in
    "download")
        get_train_data $task_date ${log_path}
        ;;
    "download_user")
        get_user_emb_data $task_date ${log_path}
        ;;
    "train")
        train_model $task_date "train" ${log_path}
        ;;
    "user_embedding")
        train_model $task_date "user_embedding" ${log_path}
        ;;
    "eval")
        train_model $task_date "test" ${log_path}
        ;;
    "predict")
        train_model $task_date "predict" ${log_path}
        ;;
    "export_new")
        train_model $task_date "export_new" ${log_path}
        ;;
    "backup")
        backup_checkpoint $task_date ${log_path}
        ;;
    "export")
        export_embedding $task_date ${log_path}
        ;;
    "save_item_embedding")
        save_item_embedding $task_date ${log_path}
        ;;
    "push_model")
        push_model $task_date ${log_path}
        ;;
    "save_item_embedding_new")
        save_item_embedding_new $task_date ${log_path}
        ;;
	"clean_data")
        clean_data $task_date ${log_path}
        ;;
    "feat_influence")
        feature_influence $task_date $receviers ${log_path}
        ;;
    "monitor")
        monitor_test_after_push_model $task_date ${log_path}
        ;;
    *)
        echo "$task_type not support!"
        ;;
esac

