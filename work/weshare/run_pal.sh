#!/bin/bash

function transfer_embedding()
{
   checkpoint=${WILL_TRANSFER_EMBEDDING_PATH}
   new_feature_path=${NEW_FEATURE_PATH}
   outpath=${OUT_EMBEDDING_PATH}
   embedding_dim=${EMB_DIM}
   CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python ${RUN_DIR}/scripts/transfer_embedding.py ${checkpoint} ${new_feature_path} ${outpath} ${embedding_dim} 
}

function _upload_cloud()
{
    src_data=$1
    des_data=$2
    log_path=$3

    retry_cnt=0
    max_retry=50
    upload_ret=0

    if [ -d ${src_data} ]; then
        upload_exe="/root/obsutil/obsutil cp -r -f"
    else
        upload_exe="/root/obsutil/obsutil cp -f"
    fi
    while [ ${retry_cnt} -lt ${max_retry} ]
    do
        ${upload_exe} ${src_data} ${des_data} 1>>${log_path}
        if [ $? -ne 0 ]; then
            ERROR_LOG "Upload ${src_data} failed, retry=${retry_cnt}" ${log_path}
            retry_cnt=`expr ${retry_cnt} + 1`
            upload_ret=1
        else
            NOTICE_LOG "Upload ${src_data} successfully" ${log_path}
            retry_cnt=0
            upload_ret=0
            break
        fi
        sleep 1
    done
    return ${upload_ret}
}

function train_model()
{
    save_dir=$1
    train_py_route=$2   # tf_trainer/main.py
    log_path=$3

    train_data_list="${save_dir}/cfg/train"
    valid_data_list="${save_dir}/cfg/valid"

    if [ ! -f ${train_data_list} ]; then
        ERROR_LOG "${train_data_list} does not exist" ${log_path}
        exit 0
    fi
    if [ ! -f ${valid_data_list} ]; then
        ERROR_LOG "${valid_data_list} does not exist" ${log_path}
        exit 0
    fi
    if [ ! -f ${save_dir}/feature_list.csv ]; then
        ERROR_LOG "${save_dir}/feature_list.csv does not exist" ${log_path}
        exit 0
    fi

    outputs_dir="${save_dir}/summaries/${MODEL}_${TASK_NAME}"

    if [ ! -d ${outputs_dir} ]; then
        mkdir -p ${outputs_dir}
    fi

    if [ ! -d ${INCREMENT_MODEL_DIR} -o "`ls -A ${INCREMENT_MODEL_DIR}`" == "" ]; then
        warm_start_dir="NULL"
    else
        warm_start_dir=${INCREMENT_MODEL_DIR}
    fi

    if [ "$ACTIVATION" == '' ]; then
        ACTIVATION="relu"
    fi

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${train_py_route} \
        --mode train \
        --model ${MODEL} \
        --network ${NETWORK} \
        --activation ${ACTIVATION} \
        --dataset ${DATASET} \
        --compression_type GZIP \
        --hidden ${DEEP_LAYERS} \
        --learning_rate ${LEARNING_RATE} \
        --epochs ${EPOCH_NUM} \
        --train_path ${train_data_list} \
        --valid_path ${valid_data_list} \
        --prebatch ${PREBATCH} \
        --batch_size ${BATCH_SIZE} \
        --checkpoint_dir ${outputs_dir} \
        --schema ${save_dir}/feature_list.csv \
        --warm_start_mode all \
        --warm_start_dir ${warm_start_dir} \
        --fixed_emb_dim ${EMB_DIM} \
        --fixed_training_steps 1000000 \
        --label_key ${LABEL_KEY} \
        --attention_cols ${ATTENTION_COLS} \
        --label_dtype ${LABEL_DTYPE} \
        ${@:2} \
        >> ${log_path} 2>&1

    if [ $? -eq 0 ]; then 
        touch ${save_dir}/train_success
    fi
}

function backup_increment_model()
{
    save_dir=$1
    log_path=$2
    version=$3  # train date

    outputs_dir="${save_dir}/summaries/${MODEL}_${TASK_NAME}"

    NOTICE_LOG "Starting backup increment model" ${log_path}

    if [ "${INCREMENT_MODEL_DIR}" == "" ]; then
        ERROR_LOG "INCREMENT_MODEL_DIR is null" ${log_path}
        exit 0
    fi
    if [ ! -d ${INCREMENT_MODEL_DIR} ]; then
        mkdir -p ${INCREMENT_MODEL_DIR}
    fi
    if [ ! -f ${outputs_dir}/checkpoint ]; then
        ERROR_LOG "${outputs_dir}/checkpoint does not exist" ${log_path}
        exit 0
    fi

    model_data=`find ${outputs_dir} -name model.*`

    if [ "${model_data}" == "" ]; then
        ERROR_LOG "${outputs_dir}/model.* does not exist" ${log_path}
        exit 0
    fi
    if [ ! -f ${outputs_dir}/graph.pbtxt ]; then
        ERROR_LOG "${outputs_dir}/graph.pbtxt does not exist" ${log_path}
        exit 0
    fi
    if [ ! -f ${outputs_dir}/checkpoint ]; then
        ERROR_LOG "${outputs_dir}/checkpoint does not exist" ${log_path}
        exit 0
    fi
    #if [ ! -d ${outputs_dir}/vocabularies/ ]; then
    #    ERROR_LOG "${outputs_dir}/vocabularies/ does not exist" ${log_path}
    #    exit 0
    #fi
    rm -f -r ${INCREMENT_MODEL_DIR}/model.* 
    cp -f ${outputs_dir}/graph.pbtxt ${INCREMENT_MODEL_DIR}/
    cp -f ${outputs_dir}/checkpoint  ${INCREMENT_MODEL_DIR}/
    cp -f ${outputs_dir}/model.*     ${INCREMENT_MODEL_DIR}/
    #cp -f -r ${outputs_dir}/vocabularies/ ${INCREMENT_MODEL_DIR}/

    if [ "${INCREMENT_MODEL_CLOUD_DIR}" != "" ]; then
        if [[ ${INCREMENT_MODEL_CLOUD_DIR} =~ /$ ]]; then
            cloud_dir=${INCREMENT_MODEL_CLOUD_DIR}${ONLINE_DICT_NAME}/${version}/
        else
            cloud_dir=${INCREMENT_MODEL_CLOUD_DIR}/${ONLINE_DICT_NAME}/${version}/
        fi
        tar -zcvPf ${save_dir}/model.tar.gz ${INCREMENT_MODEL_DIR}/ >> ${log_path} 2>&1
        _upload_cloud ${save_dir}/model.tar.gz ${cloud_dir} ${log_path}
        if [ $? -eq 0 ]; then
            rm -f ${save_dir}/model.tar.gz
        fi
    fi
    
    NOTICE_LOG "Finished backup increment model" ${log_path}
}

function monitor_eval()
{
  model_date=$1
  log_path=$2
  work_dir=$3
  valid_flag=$4
  key_type=$5

  save_path="${work_dir}/monitor_${key_type}.txt"
  rm -f ${save_path}

  if [[ ${BASE_ONLINE_DICT_LOCAL_DIR} =~ /$ ]]; then
      local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}${model_date}
  else
      local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}/${model_date}
  fi
  dict_txt="${local_dict_dir}/${ONLINE_DICT_NAME}.txt"

  feature_num=`wc -l ${dict_txt} | awk '{print $1}'`
  model_size=`du -b ${local_dict_dir}/variables | awk '{print $1}'`

  echo "[monitor] model_size=${model_size}"   >> ${save_path}
  echo "[monitor] feature_num=${feature_num}" >> ${save_path}

  log_info=`grep "Saving dict for global step" ${log_path} | tail -n 1 | sed s/[[:space:]]//g | awk -F ':|,' '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`
  for item in ${log_info}
  do
      echo "[monitor] ${item}" >> ${save_path}
  done
  base_key="SPRS.Model.daily.${ONLINE_DICT_NAME}.${key_type}"

  NOTICE_LOG "Starting importing monitor of ${model_date}" ${log_path}
  cat ${save_path} | python3 ${RUN_DIR}/scripts/import.py "${base_key}" "${model_date}" >> ${log_path} 2>&1
  NOTICE_LOG "Finished importing monitor of ${model_date}, ret=${is_model_valid}" ${log_path}
}

function monitor()
{
    model_date=$1
    log_path=$2
    work_dir=$3
    valid_flag=$4
    key_type=$5

    is_model_valid=0

    save_path="${work_dir}/monitor_${key_type}.txt"
    rm -f ${save_path}

    if [[ ${BASE_ONLINE_DICT_LOCAL_DIR} =~ /$ ]]; then
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}${model_date}
    else
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}/${model_date}
    fi
    dict_txt="${local_dict_dir}/${ONLINE_DICT_NAME}.txt"
    
    feature_num=`wc -l ${dict_txt} | awk '{print $1}'`
    model_size=`du -b ${local_dict_dir}/variables | awk '{print $1}'`

    echo "[monitor] model_size=${model_size}"   >> ${save_path}
    echo "[monitor] feature_num=${feature_num}" >> ${save_path}

    ctr=0
    auc=0
    gauc=0
    loss=0
    norm_loss=0
    calibration=0

    log_info=`grep "${valid_flag}" ${log_path} | awk '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`

    for item in ${log_info}
    do
        echo "[monitor] ${item}" >> ${save_path}
        key=`echo "${item}" | awk -F '=' '{print $1}'`
        val=`echo "${item}" | awk -F '=' '{print $2}'`
        if [ "${key}" == "ctr" ]; then
            ctr=${val}
        fi
        if [ "${key}" == "auc" ]; then
            auc=${val}
        fi
        if [ "${key}" == "gauc" ]; then
            gauc=${val}
        fi
        if [ "${key}" == "loss" ]; then
            loss=${val}
        fi
        if [ "${key}" == "norm_loss" ]; then
            norm_loss=${val}
        fi
        if [ "${key}" == "calibration" ]; then
            calibration=${val}
        fi
    done

    model_size_thresh=`echo ${VER_THRESH_MODEL_SIZE} | awk '{print $0 * 1024 * 1024}'`

    if [ $(FLOAT_LT ${auc} ${VER_THRESH_AUC}) -ne 0 ]; then
        is_model_valid=1
    elif [ $(FLOAT_LT ${gauc} ${VER_THRESH_GAUC}) -ne 0 ]; then
        is_model_valid=2
    elif [ $(FLOAT_GT ${calibration} ${VER_THRESH_CALIBRATION_UPPER}) -ne 0 ]; then
        is_model_valid=3
    elif [ $(FLOAT_LT ${calibration} ${VER_THRESH_CALIBRATION_LOWER}) -ne 0 ]; then
        is_model_valid=4
    elif [ $(FLOAT_LT ${model_size} ${model_size_thresh}) -ne 0 ]; then
        is_model_valid=5
    else
        is_model_valid=0
    fi

    base_key="SPRS.Model.daily.${ONLINE_DICT_NAME}.${key_type}"
	
    NOTICE_LOG "Starting importing monitor of ${model_date}" ${log_path}
    cat ${save_path} | python3 ${RUN_DIR}/scripts/import.py "${base_key}" "${model_date}" >> ${log_path} 2>&1
    NOTICE_LOG "Finished importing monitor of ${model_date}, ret=${is_model_valid}" ${log_path}
    return ${is_model_valid}
}

function push_model()
{
    version=$1  # train date
    log_path=$2

    if [[ ${BASE_ONLINE_DICT_LOCAL_DIR} =~ /$ ]]; then
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}${version}
    else
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}/${version}
    fi
    
    # copy model to online dict dir
    dict_txt="${local_dict_dir}/${ONLINE_DICT_NAME}.txt"
    dict_md5="${local_dict_dir}/${ONLINE_DICT_NAME}.md5"

    if [[ ${BASE_ONLINE_DICT_DIR} =~ /$ ]]; then
        online_dict_dir="${BASE_ONLINE_DICT_DIR}${ONLINE_DICT_NAME}/${version}"
    else
        online_dict_dir="${BASE_ONLINE_DICT_DIR}/${ONLINE_DICT_NAME}/${version}"
    fi

    NOTICE_LOG "online_dict_dir=${online_dict_dir}" ${log_path}
    NOTICE_LOG "local_dict_dir=${local_dict_dir}"   ${log_path}

    NOTICE_LOG "Starting pushing model to HuaweiCloud" ${log_path}

    if [ ! -f ${local_dict_dir}/saved_model.pb ]; then
        ERROR_LOG "Cannot not find ${local_dict_dir}/saved_model.pb" ${log_path}
        exit 0
    fi
    if [ ! -d ${local_dict_dir}/assets.extra ]; then
        ERROR_LOG "Cannot not find ${local_dict_dir}/assets.extra" ${log_path}
        exit 0
    fi
    if [ ! -d ${local_dict_dir}/variables ]; then
        ERROR_LOG "Cannot not find ${local_dict_dir}/variables" ${log_path}
        exit 0
    fi
    if [ ! -f ${dict_txt} ]; then
        ERROR_LOG "Cannot not find ${dict_txt}" ${log_path}
        exit 0
    fi
    if [ ! -f ${dict_md5} ]; then
        ERROR_LOG "Cannot not find ${dict_md5}" ${log_path}
        exit 0
    fi
    
    _upload_cloud ${local_dict_dir}/variables "${online_dict_dir}/" ${log_path}
    if [ $? -ne 0 ]; then
        /root/obsutil/obsutil rm -f -r ${online_dict_dir} 1>>${log_path}
        exit 0
    fi

    _upload_cloud ${local_dict_dir}/assets.extra "${online_dict_dir}/" ${log_path}
    if [ $? -ne 0 ]; then
        /root/obsutil/obsutil rm -f -r ${online_dict_dir} 1>>${log_path}
        exit 0
    fi

    _upload_cloud ${dict_txt} "${online_dict_dir}/" ${log_path}
    if [ $? -ne 0 ]; then
        /root/obsutil/obsutil rm -f -r ${online_dict_dir} 1>>${log_path}
        exit 0
    fi

    _upload_cloud ${local_dict_dir}/saved_model.pb "${online_dict_dir}/" ${log_path}
    if [ $? -ne 0 ]; then
        /root/obsutil/obsutil rm -f -r ${online_dict_dir} 1>>${log_path}
        exit 0
    fi

    _upload_cloud ${dict_md5} "${online_dict_dir}/" ${log_path}
    if [ $? -ne 0 ]; then
        /root/obsutil/obsutil rm -f -r ${online_dict_dir} 1>>${log_path}
        exit 0
    fi

    vectors_cnt=0
    for lang in hi te kn ta other; do
      if [ -f ${local_dict_dir}/${lang}_vectors.txt ]; then
        _upload_cloud ${local_dict_dir}/${lang}_vectors.txt "${online_dict_dir}/" ${log_path}
        vectors_cnt=$(($vectors_cnt+1))
      fi
    done
    if [ $vectors_cnt -gt 0 ]; then
      _upload_cloud ${local_dict_dir}/_SUCCESS "${online_dict_dir}/" ${log_path}
    fi

    NOTICE_LOG "Finished pushing model to HuaweiCloud" ${log_path}

    rm_date=`date -d "${version} -1days" +"%Y%m%d"`
    if [[ ${BASE_ONLINE_DICT_DIR} =~ /$ ]]; then
        rm_online_model="${BASE_ONLINE_DICT_DIR}${ONLINE_DICT_NAME}/${rm_date}"
    else
        rm_online_model="${BASE_ONLINE_DICT_DIR}/${ONLINE_DICT_NAME}/${rm_date}"
    fi
    /root/obsutil/obsutil rm -f -r ${rm_online_model} 1>>${log_path}

    NOTICE_LOG "Delete online_model=${rm_online_model}" ${log_path}
}

function collect_model()
{
    version=$1  # train date
    ssd_dir=$2
    log_path=$3

    feature_index_path="${ssd_dir}/${version}/feature_index/"

    NOTICE_LOG "Starting collecting model of ${version}" ${log_path}

    if [[ ${BASE_ONLINE_DICT_LOCAL_DIR} =~ /$ ]]; then
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}${version}
    else
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}/${version}
    fi
    rm -rf ${local_dict_dir}
    mkdir -p ${local_dict_dir}
    cp -r ${WORK_DIR}/${version}/summaries/${MODEL}_${TASK_NAME}/export/best/*/* ${local_dict_dir}
    if [ ! -f "${local_dict_dir}/saved_model.pb" ]; then
        ERROR_LOG "Cannot find saved_model.pb in ${local_dict_dir}" ${log_path}
        exit 0
    fi

    # copy model to online dict dir
    dict_txt="${local_dict_dir}/${ONLINE_DICT_NAME}.txt"
    dict_md5="${local_dict_dir}/${ONLINE_DICT_NAME}.md5"

    feature_map_file=${RUN_DIR}/data/${FEATURE_MAP_NAME}

    cat ${LOCAL_MODEL_SCHEMA} | awk -F',' '{if ($6==1) print $1"\t"tolower($2)"\t"$3"\t"$4"\t"$5}' > ${dict_txt}

    if [ ! -f ${dict_txt} ]; then
        ERROR_LOG "Generate ${dict_txt} failed" ${log_path}
        exit 0
    fi
    echo -e "dnn_version\t${version}" >> ${dict_txt}
    md5sum ${dict_txt} | awk '{print $1}' >> ${dict_md5}

    if [ -d "${WORK_DIR}/vectors" ]; then
        ERROR_LOG "Collecting items vector"
        cp ${WORK_DIR}/vectors/* ${local_dict_dir}
    fi

    NOTICE_LOG "Finished collecting model of ${version}" ${log_path}
}

function cdn_download()
{
    date=$1
    log_path=$2

    NOTICE_LOG "Starting downloading data" ${log_path}
    if [[ ${BASE_CDN_DATA_DIR} =~ /$ ]]; then
        python3 ${RUN_DIR}/scripts/download.py ${BASE_CDN_DATA_DIR}${date}  >> ${log_path} 2>&1
    else
        python3 ${RUN_DIR}/scripts/download.py ${BASE_CDN_DATA_DIR}/${date} >> ${log_path} 2>&1
    fi
    NOTICE_LOG "Finished downloading data" ${log_path}
}

function huawei_download()
{
    date=$1
    log_path=$2
    mhd_dir=$3

    if [[ ${BASE_CDN_DATA_DIR} =~ /$ ]]; then
        data_dir=${BASE_CDN_DATA_DIR}${date}
    else
        data_dir=${BASE_CDN_DATA_DIR}/${date}
    fi

    while true
    do
        /root/obsutil/obsutil stat ${data_dir}/pred_data/_SUCCESS
        if [ $? -ne 0 ]; then
            NOTICE_LOG "${data_dir} is not ready yet" ${log_path}
            sleep 300
        else
            break
    fi
    done

    NOTICE_LOG "Starting downloading data" ${log_path}
    /root/obsutil/obsutil cp -f -r ${data_dir} ${mhd_dir} 1>>${log_path}

    task_id=""
    retry_cnt=0
    max_retry=50

    while true
    do
        if [ ${retry_cnt} -lt ${max_retry} ]; then
            retry_cnt=`expr ${retry_cnt} + 1`
        else
            break
        fi

        fail_count=`grep "Succeed count is:" ${log_path} | tail -1 | awk '{print $8}'`
        if [ ${fail_count} -ne 0  ]; then
            task_id=`grep "Task id is:" ${log_path} | tail -1 | awk '{print $4}'`
            NOTICE_LOG "task_id=${task_id}, retry=${retry_cnt}" ${log_path}
            /root/obsutil/obsutil cp -recover=${task_id} -f -r 1>>${log_path}
        else
            break
        fi
        sleep 1
    done
    NOTICE_LOG "Finished downloading data" ${log_path}
}

function prepare_train()
{
    date=$1
    ssd_dir=$2
    work_dir=$3
    log_path=$4

    NOTICE_LOG "Starting generating feature map" ${log_path}

    feature_lst_path=${work_dir}/feature_list.csv
    feature_map_file=${RUN_DIR}/data/${FEATURE_MAP_NAME}

    cat ${LOCAL_MODEL_SCHEMA} > ${feature_lst_path}
    NOTICE_LOG "Finished generating feature map" ${log_path}
    
    NOTICE_LOG "Starting generating training cfg" ${log_path}
    python3 ${RUN_DIR}/scripts/gen_cfg.py ${ssd_dir}/${date} ${work_dir} 2>>${log_path}
    NOTICE_LOG "Finished generating training cfg" ${log_path}

    train_data_list="${work_dir}/cfg/train"
    valid_data_list="${work_dir}/cfg/valid"

    trn_list_size=`wc -l ${train_data_list} | awk '{print $1}'`
    val_list_size=`wc -l ${valid_data_list} | awk '{print $1}'`
    if [ ${trn_list_size} -eq 0 -o ${val_list_size} -eq 0 ]; then
        ERROR_LOG "${train_data_list} or ${valid_data_list} is null" ${log_path}
        exit 0
    fi
    cp ${valid_data_list} ${work_dir}/valid_data_list
}

function train()
{
    work_dir=$1
    log_path=$2

    train_py_route="./main.py"

    NOTICE_LOG "Starting training model" ${log_path}
    train_model ${work_dir} ${train_py_route} ${log_path}
    NOTICE_LOG "Finished training model" ${log_path}
}

function eval()
{
    work_dir=$1
    log_path=$2
    model_date=$3
    feature_name=$4

    pred_path="${work_dir}/pred.txt"
    model_dir="${WORK_DIR}/${model_date}/summaries/${MODEL}_${TASK_NAME}"

    valid_data_list="${work_dir}/valid_data_list"
    feature_list_path="${work_dir}/feature_list.csv"

    if [ ! -d ${model_dir} ]; then
        ERROR_LOG "Cannot find model_dir=${model_dir}" ${log_path}
        exit 0
    fi
    if [ ! -f ${valid_data_list} ]; then
        ERROR_LOG "Cannot find valid_data_list=${valid_data_list}" ${log_path}
        exit 0
    fi
    if [ ! -f ${feature_list_path} ]; then
        ERROR_LOG "Cannot find feature_list_path=${feature_lst_path}" ${log_path}
        exit 0
    fi

    predict_py_route="./main.py"
    if [ "$feature_name" == '' ]; then
        feature_name='None'
        NOTICE_LOG "Starting eval with all feature" ${log_path}
    else
        NOTICE_LOG "Starting eval without feature $feature_name" ${log_path}
    fi

    if [ "$ACTIVATION" == '' ]; then
        ACTIVATION="relu"
    fi

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${predict_py_route} \
        --mode test \
        --attention_cols ${ATTENTION_COLS} \
        --schema ${feature_list_path} \
        --feature_influence $feature_name \
        --model ${MODEL} \
        --network ${NETWORK} \
        --activation ${ACTIVATION} \
        --dataset ${DATASET} \
        --compression_type GZIP \
        --hidden ${DEEP_LAYERS} \
        --learning_rate ${LEARNING_RATE} \
        --epochs ${EPOCH_NUM} \
        --train_path ${valid_data_list} \
        --valid_path ${valid_data_list} \
        --prebatch ${PREBATCH} \
        --batch_size ${BATCH_SIZE} \
        --checkpoint_dir ${model_dir} \
        --fixed_emb_dim ${EMB_DIM} \
        --fixed_training_steps 1000000 \
        --label_key ${LABEL_KEY} \
        --pred_path ${pred_path} \
        --debug_labels ${DEBUG_LABELS} \
        --label_dtype ${LABEL_DTYPE} \
        >> ${log_path} 2>&1

}

function _influence_out() {
  log_path=$1
  influence_file=$2
  feature_name=$3
  log_info=`grep "Saving dict for global step" ${log_path} | tail -n 1 | sed s/[[:space:]]//g | awk -F ':|,' '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`
  for item in ${log_info}
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

function feature_influence()
{
    work_dir=$1
    log_path=$2
    model_date=$3
    influence="${work_dir}/influence.csv"
    rm -f $influence
    echo "name,auc,loss,score" > $influence
    NOTICE_LOG "Starting feature influence" ${log_path}
    eval ${work_dir} ${log_path} ${model_date}
    _influence_out $log_path $influence "base"

    feature_list_path="${work_dir}/feature_list.csv"
    features=`cat $feature_list_path | awk -F ',' '{print $1}'`
    for feat in ${features}
    do
      eval ${work_dir} ${log_path} ${model_date} $feat
      _influence_out $log_path $influence $feat
    done
    NOTICE_LOG "Done feature influence" ${log_path}
}

function get_item_embedding()
{
    work_dir=$1
    log_path=$2
    model_date=$3

    items_obs_path="obs://sprs-data-sg/sharezone/recall/item_feature/datepart=$model_date"
    task_id=""
    retry_cnt=0
    max_retry=50

    /root/obsutil/obsutil cp -f -r ${items_obs_path} ${work_dir}/recall_item_data 1>>${log_path}
    while true
    do
        if [ ${retry_cnt} -lt ${max_retry} ]; then
            retry_cnt=`expr ${retry_cnt} + 1`
        else
            break
        fi
        fail_count=`grep "Succeed count is:" ${log_path} | tail -1 | awk '{print $8}'`
        if [ ${fail_count} -ne 0  ]; then
            task_id=`grep "Task id is:" ${log_path} | tail -1 | awk '{print $4}'`
            NOTICE_LOG "task_id=${task_id}, retry=${retry_cnt}" ${log_path}
            /root/obsutil/obsutil cp -recover=${task_id} -f -r 1>>${log_path}
        else
            break
        fi
        sleep 1
    done


    pred_path="${work_dir}/embeddings.txt"
    model_dir="${WORK_DIR}/${model_date}/summaries/${MODEL}_${TASK_NAME}"

    valid_data_list="${work_dir}/recall_item_data/datepart=${model_date}/test_data"
    feature_list_path="${work_dir}/feature_list.csv"

    if [ ! -d ${model_dir} ]; then
        ERROR_LOG "Cannot find model_dir=${model_dir}" ${log_path}
        exit 0
    fi
    if [ ! -d ${valid_data_list} ]; then
        ERROR_LOG "Cannot find valid_data_list=${valid_data_list}" ${log_path}
        exit 0
    fi
    if [ ! -f ${feature_list_path} ]; then
        ERROR_LOG "Cannot find feature_list_path=${feature_lst_path}" ${log_path}
        exit 0
    fi

    if [ "$ACTIVATION" == '' ]; then
        ACTIVATION="relu"
    fi

    batch_size=$((${PREBATCH}*${BATCH_SIZE}))
    predict_py_route="./main.py"

    NOTICE_LOG "predict_py_route=${predict_py_route}" ${log_path}
    NOTICE_LOG "Starting get embedding with model=${model_dir}" ${log_path}

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${predict_py_route} \
        --mode predict \
        --attention_cols ${ATTENTION_COLS} \
        --schema ${feature_list_path} \
        --model ${MODEL} \
        --network ${NETWORK} \
        --activation ${ACTIVATION} \
        --dataset ${DATASET} \
        --compression_type GZIP \
        --hidden ${DEEP_LAYERS} \
        --learning_rate ${LEARNING_RATE} \
        --epochs ${EPOCH_NUM} \
        --train_path ${valid_data_list} \
        --valid_path ${valid_data_list} \
        --prebatch 1 \
        --batch_size ${batch_size} \
        --checkpoint_dir ${model_dir} \
        --fixed_emb_dim ${EMB_DIM} \
        --fixed_training_steps 1000000 \
        --label_key ${LABEL_KEY} \
        --pred_path ${pred_path} \
        --debug_labels ${DEBUG_LABELS} \
        --label_dtype ${LABEL_DTYPE} \
        >> ${log_path} 2>&1

    item_index_file="${work_dir}/recall_item_data/datepart=${model_date}/item_index/part-00000"
    index_embedding_file="${work_dir}/vectors/"
    python3 ${RUN_DIR}/scripts/item_embedding.py $item_index_file $pred_path $index_embedding_file
    NOTICE_LOG "Finished get embedding with model=${model_dir}" ${log_path}
}

function predict()
{
    work_dir=$1
    log_path=$2
    valid_flag=$3
    model_date=$4

    pred_path="${work_dir}/pred.txt"
    model_dir="${WORK_DIR}/${model_date}/summaries/${MODEL}_${TASK_NAME}"

    valid_data_list="${work_dir}/valid_data_list"
    feature_list_path="${work_dir}/feature_list.csv"

    if [ ! -d ${model_dir} ]; then
        ERROR_LOG "Cannot find model_dir=${model_dir}" ${log_path}
        exit 0
    fi
    if [ ! -f ${valid_data_list} ]; then
        ERROR_LOG "Cannot find valid_data_list=${valid_data_list}" ${log_path}
        exit 0
    fi
    if [ ! -f ${feature_list_path} ]; then
        ERROR_LOG "Cannot find feature_list_path=${feature_lst_path}" ${log_path}
        exit 0
    fi

    if [ "$ACTIVATION" == '' ]; then
        ACTIVATION="relu"
    fi

    predict_py_route="./main.py"

    NOTICE_LOG "predict_py_route=${predict_py_route}" ${log_path}
    NOTICE_LOG "Starting predicting with model=${model_dir}" ${log_path}

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 ${predict_py_route} \
        --mode predict \
        --attention_cols ${ATTENTION_COLS} \
        --schema ${feature_list_path} \
        --model ${MODEL} \
        --network ${NETWORK} \
        --activation ${ACTIVATION} \
        --dataset ${DATASET} \
        --compression_type GZIP \
        --hidden ${DEEP_LAYERS} \
        --learning_rate ${LEARNING_RATE} \
        --epochs ${EPOCH_NUM} \
        --train_path ${valid_data_list} \
        --valid_path ${valid_data_list} \
        --prebatch ${PREBATCH} \
        --batch_size ${BATCH_SIZE} \
        --checkpoint_dir ${model_dir} \
        --fixed_emb_dim ${EMB_DIM} \
        --fixed_training_steps 1000000 \
        --label_key ${LABEL_KEY} \
        --pred_path ${pred_path} \
        --debug_labels ${DEBUG_LABELS} \
        --label_dtype ${LABEL_DTYPE} \
        >> ${log_path} 2>&1

    NOTICE_LOG "Finished predicting with model=${model_dir}" ${log_path}

    NOTICE_LOG "Starting calculating metrics" ${log_path}
    is_user_type=`echo ${PRED_OUTPUT_FORMAT} | grep "user_type"`
    if [ "${is_user_type}" != "" ]; then
        cat ${pred_path} | python3 ${RUN_DIR}/scripts/auc.py main_user_type ${PRED_OUTPUT_FORMAT} ${valid_flag} >> ${log_path} 2>&1
    else
        cat ${pred_path} | python3 ${RUN_DIR}/scripts/auc.py main ${PRED_OUTPUT_FORMAT} ${valid_flag} >> ${log_path} 2>&1
    fi
    NOTICE_LOG "Finished calculating metrics" ${log_path}
}

function clean_data()
{
    date=$1
    ssd_dir=$2
    mhd_dir=$3

    if [ "${WORK_ON_SSD_MHD}" == "ssd" ]; then
        # clean train data
        rm_date=`date -d "${date} -5days" +"%Y%m%d"`
        rm -rf ${ssd_dir}/${rm_date}
        NOTICE_LOG "Delete data=${ssd_dir}/${rm_date}" ${log_path}

        # clean local_online_dict
        rm_date=`date -d "${date} -5days" +"%Y%m%d"`
        rm -rf ${BASE_ONLINE_DICT_LOCAL_DIR}/${rm_date}
        NOTICE_LOG "Delete online_dict=${BASE_ONLINE_DICT_LOCAL_DIR}/${rm_date}" ${log_path}

        # clean local_work_dir
        rm_date=`date -d "${date} -5days" +"%Y%m%d"`
        rm -rf ${WORK_DIR}/${rm_date}
        NOTICE_LOG "Delete work_dir=${WORK_DIR}/${rm_date}" ${log_path}

    else
        # clean train data
        rm_date=`date -d "${date} -7days" +"%Y%m%d"`
        rm -rf ${mhd_dir}/${rm_date}
        NOTICE_LOG "Delete data=${mhd_dir}/${rm_date}" ${log_path}

        # clean local_online_dict
        rm_date=`date -d "${date} -7days" +"%Y%m%d"`
        rm -rf ${BASE_ONLINE_DICT_LOCAL_DIR}/${rm_date}
        NOTICE_LOG "Delete online_dict=${BASE_ONLINE_DICT_LOCAL_DIR}/${rm_date}" ${log_path}

        # clean local_work_dir
        rm_date=`date -d "${date} -7days" +"%Y%m%d"`
        rm -rf ${WORK_DIR}/${rm_date}
        NOTICE_LOG "Delete work_dir=${WORK_DIR}/${rm_date}" ${log_path}
    fi
}

function send_mail()
{
    date=$1
    work_dir=$2
    log_path=$3

    NOTICE_LOG "Starting sending mail for model updating" ${log_path}
    if [ ! -f ${work_dir}/monitor_test.txt ]; then
        ERROR_LOG "Cannot find ${work_dir}/monitor_test.txt" ${log_path}
        exit 0
    fi
    subject="${MAIL_SUBJECT} ${date}"
    msg="<p>Congratulations! DNN model is updated successfully. ^o^</p>"
    msg="${msg}<p></p>Key indicators as follows<br>"
    monitor_msg=`awk -F ' ' '{print "----> "$2"<br>"}' ${work_dir}/monitor_test.txt`
    msg="${msg}${monitor_msg}"
    python3 ${RUN_DIR}/scripts/send_mail.py "${MAIL_ADDRESS}" "${subject}" "${msg}"
    NOTICE_LOG "Finished sending mail for model updating" ${log_path}
}

function check_config()
{
    if [ "${LABEL_KEY}" == "" ]; then
        LABEL_KEY="is_click"
    fi
    if [ "${LABEL_DTYPE}" == "" ]; then
        LABEL_DTYPE="float"
    fi
    if [ "${USE_SEQUENCE_WEIGHT}" == "" ]; then
        USE_SEQUENCE_WEIGHT="False"
    fi
    if [ "${EMB_DIM}" == "" ]; then
        EMB_DIM=8
    fi
    if [ "${DNN_WITH_HASH}" == "" ]; then
        DNN_WITH_HASH="True"
    fi
    if [ "${PRED_OUTPUT_FORMAT}" == "" ]; then
        PRED_OUTPUT_FORMAT="label,user_id,pctr"
    fi
    if [ "${USE_INCREMENT_MODE}" == "" ]; then
        USE_INCREMENT_MODE="False"
    fi
    if [ "${USE_INCREMENT_MODE}" == "False" ]; then
        INCREMENT_MODEL_DIR=""
    fi
    if [ "${ATTENTION_COLS}" == "" ]; then
        ATTENTION_COLS="NULL"
    fi
    if [ "${IS_POSTERIOR_VALID}" == "" ]; then
        IS_POSTERIOR_VALID="False"
    fi
    if [ "${PIPELINE_CLEAN_DATA}" == "" ]; then
        PIPELINE_CLEAN_DATA="False"
    fi
}

function pipeline2()
{
    date=$1
    last_date=$2
    ssd_save_dir=$3
    work_dir=$4
    log_path=$5

    # check data
    if [ ! -d ${ssd_save_dir}/${date} ]; then
        ERROR_LOG "Data=${ssd_save_dir}/${date} does not exist" ${log_path}
        exit 0
    fi

    # prepara_train
    prepare_train ${date} ${ssd_save_dir} ${work_dir} ${log_path}

    # train
    train ${work_dir} ${log_path}

    # backup model
    if [ "${USE_INCREMENT_MODE}" == "True" ]; then
        backup_increment_model ${work_dir} ${log_path} ${date}
    fi

    # collect_model
    collect_model ${date} ${ssd_save_dir} ${log_path}

    if [ "${DO_EVAL}" == 'True' ]; then
      if [ "${IS_POSTERIOR_VALID}" == "False" ]; then
          # eval
          eval ${work_dir} ${log_path} ${date}

          # monitor
          monitor_eval ${date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"

          error=$?
          if [ ${error} -gt 0 ]; then
              ERROR_LOG "Model is illegal, error code is ${error}" ${log_path}
              exit 0
          fi
      else
          # eval
          eval ${work_dir} ${log_path} ${date}

          # monitor
          monitor_eval ${date} ${log_path} ${work_dir} ${TRN_VALID_FLAG} "train"

          # eval
          eval ${work_dir} ${log_path} ${last_date}

          # monitor
          monitor_eval ${last_date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"

          error=$?
          if [ ${error} -gt 0 ]; then
              ERROR_LOG "Model is illegal, error code is ${error}" ${log_path}
              exit 0
          fi
      fi
    else
      if [ "${IS_POSTERIOR_VALID}" == "False" ]; then
          # predict
          predict ${work_dir} ${log_path} ${TST_VALID_FLAG} ${date}

          # monitor
          monitor ${date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"

          error=$?
          if [ ${error} -gt 0 ]; then
              ERROR_LOG "Model is illegal, error code is ${error}" ${log_path}
              exit 0
          fi
      else
          # predict with current datepart model
          predict ${work_dir} ${log_path} ${TRN_VALID_FLAG} ${date}

          # monitor current model with train data
          monitor ${date} ${log_path} ${work_dir} ${TRN_VALID_FLAG} "train"

          # predict with last datepart model on current train data
          predict ${work_dir} ${log_path} ${TST_VALID_FLAG} ${last_date}

          # monitor last model with train data
          monitor ${last_date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"

          error=$?
          if [ ${error} -gt 0 ]; then
              ERROR_LOG "Model is illegal, error code is ${error}" ${log_path}
              exit 0
          fi
      fi
    fi

    # push_online
    push_model ${date} ${log_path}

    # send mail
    send_mail ${date} ${work_dir} ${log_path}

    if [ ${PIPELINE_CLEAN_DATA} == "True" ]; then
        # clean data
        clean_data ${date} ${ssd_save_dir} ${mhd_save_dir}
    fi
}

if [ $# -lt 2 ]; then
    echo "Usage: $0 [conf_file] [task_type] [date]"
    echo "[date] is optional, if missing, set it to yesterday"
    echo "[task_type] options as following"
    echo "-----   <cdn_index>          --  Generate download index with aws server machine."
    echo "-----   <download>           --  Download data from s3/obs cdn."
    echo "-----   <prepare_train>      --  Generate cfg/feature_map etc. for training."
    echo "-----   <train>              --  Train model."
    echo "-----   <eval>               --  Eval model."
    echo "-----   <backup_model>       --  Backup model for incremental trainning."
    echo "-----   <predict>            --  Predict data with the model of [date]."
    echo "-----   <feat_influence>     --  Eval feature influence of [date]."
    echo "-----   <collect>            --  Collect offline model for online use."
    echo "-----   <monitor>            --  Check model and upload monitor indicators."
    echo "-----   <monitor_eval>       --  Check model and upload monitor indicators."
    echo "-----   <send_mail>          --  Send mail for updating new model."
    echo "-----   <transfer_embedding> --  Transfer feature embedding to new file"
    echo "-----   <push_online>        --  Push model to online server."
    echo "-----   <get_item_embedding>--  Push item embedding to online faiss"
    echo "-----   <clean_data>         --  Clean expired data."
    echo "-----   <pipeline2>          --  Execute a series of operations to update model except <download>."
    echo "-----   <pipeline1>          --  Execute <download>,<pipeline2>."
    echo -e "e.g. $0 conf/config.sh train"
    echo -e "e.g. $0 conf/config.sh train 20190730\n"
    exit 0
fi

conf_file=$1

if [ ! -f ${conf_file} ]; then
    echo "[ERROR] ${conf_file} does not exist"
    exit 0
fi

source ${conf_file}

if [ ! -f ${RUN_DIR}/scripts/util.sh ]; then
    echo "[ERROR] ${RUN_DIR}/scripts/util.sh does not exist"
    exit 0
fi

source ${RUN_DIR}/scripts/util.sh

task_type=$2

date=`date -d"-1days" +%Y%m%d`
if [ $# -eq 3 ]; then
    date=$3
fi

last_date=`date -d "${date} -1days" +"%Y%m%d"`

work_dir="${WORK_DIR}/${date}"
if [ ! -d ${work_dir} ]; then
    mkdir -p ${work_dir}
fi
log_path="${work_dir}/run.log"
NOTICE_LOG "work_dir=${work_dir}" ${log_path}

if [[ ${BASE_CDN_DATA_DIR} =~ /$ ]]; then
    share_sub_dir=`echo "${BASE_CDN_DATA_DIR}" | awk -F '/' '{print $(NF-2)"/"$(NF-1)}'`
else
    share_sub_dir=`echo "${BASE_CDN_DATA_DIR}" | awk -F '/' '{print $(NF-1)"/"$NF}'`
fi

if [ "${WORK_ON_SSD_MHD}" == "ssd" ]; then
    ssd_save_dir="${SSD_PATH}/${share_sub_dir}/"
    if [ ! -d ${ssd_save_dir} ]; then
        mkdir -p ${ssd_save_dir}
    fi
    mhd_save_dir=${ssd_save_dir}
elif [ "${WORK_ON_SSD_MHD}" == "mhd" ]; then
    mhd_save_dir="${MHD_PATH}/${share_sub_dir}/"
    if [ ! -d ${mhd_save_dir} ]; then
        mkdir -p ${mhd_save_dir}
    fi
    ssd_save_dir=${mhd_save_dir}
else
    ERROR_LOG "Illegal WORK_ON_SSD_MHD=${WORK_ON_SSD_MHD}" ${log_path}
    exit 0
fi

export TRN_VALID_FLAG="Final-train-valid"
export TST_VALID_FLAG="Final-test-valid"

NOTICE_LOG "mhd_save_dir=${mhd_save_dir}" ${log_path}
NOTICE_LOG "ssd_save_dir=${ssd_save_dir}" ${log_path}

check_config

if [ "${task_type}" == "cdn_index" ]; then
    aws s3 cp s3://recommendation.ap-southeast-1/sharezone_cdn/index.py ./
    if [ ! -f index.py ]; then
        ERROR_LOG "Download index.py failed" ${log_path}
        exit 0
    fi
    NOTICE_LOG "Starting generating index" ${log_path}
    python3 index.py ${BASE_CDN_DATA_DIR}/${date}
    NOTICE_LOG "Finished generating index" ${log_path}

elif [ "${task_type}" == "download" ]; then
    #cdn_download ${date} ${log_path}
    huawei_download ${date} ${log_path} ${mhd_save_dir}

    if [ "${mhd_save_dir}" != "${ssd_save_dir}" ]; then
        NOTICE_LOG "Starting copying data from mhd to ssd" ${log_path}
        cp -r ${mhd_save_dir}/${date} ${ssd_save_dir}
        NOTICE_LOG "Finished copying data from mhd to ssd" ${log_path}
    fi

elif [ "${task_type}" == "transfer_embedding" ]; then
     transfer_embedding 
     
elif [ "${task_type}" == "prepare_train" ]; then
    prepare_train ${date} ${ssd_save_dir} ${work_dir} ${log_path}

elif [ "${task_type}" == "train" ]; then
    train ${work_dir} ${log_path}

elif [ "${task_type}" == "eval" ]; then
    if [ "${IS_POSTERIOR_VALID}" == "False" ]; then
        eval ${work_dir} ${log_path} ${date}
    else
        eval ${work_dir} ${log_path} ${date}
        eval ${work_dir} ${log_path} ${last_date}
    fi

elif [ "${task_type}" == "predict" ]; then
    if [ "${IS_POSTERIOR_VALID}" == "False" ]; then
        predict ${work_dir} ${log_path} ${TST_VALID_FLAG} ${date}
    else
        predict ${work_dir} ${log_path} ${TRN_VALID_FLAG} ${date}
        predict ${work_dir} ${log_path} ${TST_VALID_FLAG} ${last_date}
    fi

elif [ "${task_type}" == "get_item_embedding" ]; then
    get_item_embedding ${work_dir} ${log_path} ${date}

elif [ "${task_type}" == "feat_influence" ]; then
    feature_influence ${work_dir} ${log_path} ${date}

elif [ "${task_type}" == "collect" ]; then
    collect_model ${date} ${ssd_save_dir} ${log_path}

elif [ "${task_type}" == "backup_model" ]; then
    backup_increment_model ${work_dir} ${log_path} ${date}

elif [ "${task_type}" == "push_online" ]; then
    push_model ${date} ${log_path}

elif [ "${task_type}" == "send_mail" ]; then
    send_mail ${date} ${work_dir} ${log_path}

elif [ "${task_type}" == "monitor" ]; then
    if [ "${IS_POSTERIOR_VALID}" == "False" ]; then
        monitor ${date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"
    else
        monitor ${date} ${log_path} ${work_dir} ${TRN_VALID_FLAG} "train"
        monitor ${last_date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"
    fi

elif [ "${task_type}" == "monitor_eval" ]; then
    if [ "${IS_POSTERIOR_VALID}" == "False" ]; then
        monitor_eval ${date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"
    else
        monitor_eval ${date} ${log_path} ${work_dir} ${TRN_VALID_FLAG} "train"
        monitor_eval ${last_date} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"
    fi

elif [ "${task_type}" == "clean_data" ]; then
    clean_data ${date} ${ssd_save_dir} ${mhd_save_dir}

elif [ "${task_type}" == "pipeline1" ]; then
    # download
    #cdn_download ${date} ${log_path}
    huawei_download ${date} ${log_path} ${mhd_save_dir}
    
    pipeline2 ${date} ${last_date} ${ssd_save_dir} ${work_dir} ${log_path}
     
elif [ "${task_type}" == "pipeline2" ]; then
    pipeline2 ${date} ${last_date} ${ssd_save_dir} ${work_dir} ${log_path}

else
    ERROR_LOG "Illegal task_type=${task_type}" ${log_path}
fi

exit 0
