#!/bin/bash

function __upload_cloud()
{
    src_data=$1
    des_data=$2
    log_path=$3

    retry_cnt=0
    max_retry=50
    upload_ret=0

    if [ -d ${src_data} ]; then
        upload_exe="${OBSUTIL} cp -r -f"
    else
        upload_exe="${OBSUTIL} cp -f"
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
    work_dir=$1
    log_path=$2

    train_py_route="./main.py"

    train_data_list="${work_dir}/cfg/train"
    valid_data_list="${work_dir}/cfg/valid"

    if [ ! -f ${train_data_list} ]; then
        ERROR_LOG "${train_data_list} does not exist" ${log_path}
        exit 1
    fi
    if [ ! -f ${valid_data_list} ]; then
        ERROR_LOG "${valid_data_list} does not exist" ${log_path}
        exit 1
    fi
    if [ ! -f ${work_dir}/feature_list.csv ]; then
        ERROR_LOG "${work_dir}/feature_list.csv does not exist" ${log_path}
        exit 1
    fi

    outputs_dir="${work_dir}/summaries/${MODEL}_${TASK_NAME}"

    if [ ! -d ${outputs_dir} ]; then
        mkdir -p ${outputs_dir}
    fi

    if [ ! -d ${INCREMENT_MODEL_DIR} -o "`ls -A ${INCREMENT_MODEL_DIR}`" == "" ]; then
        warm_start_dir="NULL"
    else
        warm_start_dir=${INCREMENT_MODEL_DIR}
    fi

    python3 ${train_py_route} \
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
        --schema ${work_dir}/feature_list.csv \
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
        touch ${work_dir}/train_success
    fi
}

function backup_increment_model()
{
    work_dir=$1
    log_path=$2
    version=$3  # train timstamp

    outputs_dir="${work_dir}/summaries/${MODEL}_${TASK_NAME}"

    NOTICE_LOG "Starting backup increment model" ${log_path}

    if [ "${INCREMENT_MODEL_DIR}" == "" ]; then
        ERROR_LOG "INCREMENT_MODEL_DIR is null" ${log_path}
        exit 1
    fi
    if [ ! -d ${INCREMENT_MODEL_DIR} ]; then
        mkdir -p ${INCREMENT_MODEL_DIR}
    fi
    if [ ! -f ${outputs_dir}/checkpoint ]; then
        ERROR_LOG "${outputs_dir}/checkpoint does not exist" ${log_path}
        exit 1
    fi

    model_data=`find ${outputs_dir} -name model.*`

    if [ "${model_data}" == "" ]; then
        ERROR_LOG "${outputs_dir}/model.* does not exist" ${log_path}
        exit 1
    fi
    if [ ! -f ${outputs_dir}/graph.pbtxt ]; then
        ERROR_LOG "${outputs_dir}/graph.pbtxt does not exist" ${log_path}
        exit 1
    fi
    if [ ! -f ${outputs_dir}/checkpoint ]; then
        ERROR_LOG "${outputs_dir}/checkpoint does not exist" ${log_path}
        exit 1
    fi

    rm -f -r ${INCREMENT_MODEL_DIR}/model.* 
    cp -f ${outputs_dir}/graph.pbtxt ${INCREMENT_MODEL_DIR}/
    cp -f ${outputs_dir}/checkpoint  ${INCREMENT_MODEL_DIR}/
    cp -f ${outputs_dir}/model.*     ${INCREMENT_MODEL_DIR}/

    if [ "${INCREMENT_MODEL_CLOUD_DIR}" != "" ]; then
        if [[ ${INCREMENT_MODEL_CLOUD_DIR} =~ /$ ]]; then
            cloud_dir=${INCREMENT_MODEL_CLOUD_DIR}${ONLINE_DICT_NAME}/${version}/
        else
            cloud_dir=${INCREMENT_MODEL_CLOUD_DIR}/${ONLINE_DICT_NAME}/${version}/
        fi
        tar -zcvPf ${work_dir}/model.tar.gz ${INCREMENT_MODEL_DIR}/ >> ${log_path} 2>&1
        __upload_cloud ${work_dir}/model.tar.gz ${cloud_dir} ${log_path}
        if [ $? -eq 0 ]; then
            rm -f ${work_dir}/model.tar.gz
        fi
    fi
    
    NOTICE_LOG "Finished backup increment model" ${log_path}
}

function monitor_eval()
{
    model_ver=$1
    log_path=$2
    work_dir=$3
    key_type=$4

    save_path="${work_dir}/monitor_${key_type}.txt"
    rm -f ${save_path}

    if [[ ${BASE_ONLINE_DICT_LOCAL_DIR} =~ /$ ]]; then
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}${model_ver}
    else
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}/${model_ver}
    fi
    dict_txt="${local_dict_dir}/${ONLINE_DICT_NAME}.txt"

    feature_num=`wc -l ${dict_txt} | awk '{print $1}'`
    model_size=`du -b ${local_dict_dir}/variables | awk '{print $1}'`

    echo "[monitor] model_size=${model_size}"   >> ${save_path}
    echo "[monitor] feature_num=${feature_num}" >> ${save_path}

    log_info=`grep "Saving dict for global step" ${log_path} | \
        tail -n 1 | sed s/[[:space:]]//g | awk -F ':|,' '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`
    for item in ${log_info}
    do
        echo "[monitor] ${item}" >> ${save_path}
    done
    base_key="SPRS.Model.daily.${ONLINE_DICT_NAME}.${key_type}"

    NOTICE_LOG "Starting importing monitor of ${model_ver}" ${log_path}
    cat ${save_path} | python3 ${RUN_DIR}/scripts/import.py "${base_key}" "${model_ver}" >> ${log_path} 2>&1
    NOTICE_LOG "Finished importing monitor of ${model_ver}" ${log_path}
    return 0
}

function monitor()
{
    model_ver=$1
    log_path=$2
    work_dir=$3
    out_flag=$4
    key_type=$5

    save_path="${work_dir}/monitor_${key_type}.txt"
    rm -f ${save_path}

    if [[ ${BASE_ONLINE_DICT_LOCAL_DIR} =~ /$ ]]; then
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}${model_ver}
    else
        local_dict_dir=${BASE_ONLINE_DICT_LOCAL_DIR}/${model_ver}
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
    positive_rate=0

    log_info=`grep "${out_flag}" ${log_path} | awk '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`

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
        if [ "${key}" == "positive_rate" ]; then
            positive_rate=${val}
        fi
    done

    model_size_thresh=`echo ${VER_THRESH_MODEL_SIZE} | awk '{print $0 * 1024 * 1024}'`

    if [ $(FLOAT_LT ${auc} ${VER_THRESH_AUC}) -ne 0 ]; then
        valid_flag=1
    elif [ $(FLOAT_LT ${gauc} ${VER_THRESH_GAUC}) -ne 0 ]; then
        valid_flag=2
    elif [ $(FLOAT_GT ${calibration} ${VER_THRESH_CALIBRATION_UPPER}) -ne 0 ]; then
        valid_flag=3
    elif [ $(FLOAT_LT ${calibration} ${VER_THRESH_CALIBRATION_LOWER}) -ne 0 ]; then
        valid_flag=4
    elif [ $(FLOAT_LT ${model_size} ${model_size_thresh}) -ne 0 ]; then
        valid_flag=5
    else
        valid_flag=0
    fi

    base_key="SPRS.Model.daily.${ONLINE_DICT_NAME}.${key_type}"
	
    NOTICE_LOG "Starting importing monitor of ${model_ver}" ${log_path}
    cat ${save_path} | python3 ${RUN_DIR}/scripts/import.py "${base_key}" "${model_ver}" >> ${log_path} 2>&1
    NOTICE_LOG "Finished importing monitor of ${model_ver}, ret=${valid_flag}" ${log_path}
    return ${valid_flag}
}

function push_model()
{
    version=$1      # train timestamp
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
        online_dict_dir="${BASE_ONLINE_DICT_DIR}${ONLINE_DICT_NAME}"
    else
        online_dict_dir="${BASE_ONLINE_DICT_DIR}/${ONLINE_DICT_NAME}"
    fi

    NOTICE_LOG "online_dict_dir=${online_dict_dir}/${version}" ${log_path}
    NOTICE_LOG "local_dict_dir=${local_dict_dir}" ${log_path}

    NOTICE_LOG "Starting pushing model to HuaweiCloud" ${log_path}

    if [ ! -f ${local_dict_dir}/saved_model.pb ]; then
        ERROR_LOG "Cannot not find ${local_dict_dir}/saved_model.pb" ${log_path}
        exit 1
    fi
    if [ ! -d ${local_dict_dir}/assets.extra ]; then
        ERROR_LOG "Cannot not find ${local_dict_dir}/assets.extra" ${log_path}
        exit 1
    fi
    if [ ! -d ${local_dict_dir}/variables ]; then
        ERROR_LOG "Cannot not find ${local_dict_dir}/variables" ${log_path}
        exit 1
    fi
    if [ ! -f ${dict_txt} ]; then
        ERROR_LOG "Cannot not find ${dict_txt}" ${log_path}
        exit 1
    fi
    if [ ! -f ${dict_md5} ]; then
        ERROR_LOG "Cannot not find ${dict_md5}" ${log_path}
        exit 1
    fi
    
    __upload_cloud ${local_dict_dir}/variables "${online_dict_dir}/${version}/" ${log_path}

    if [ $? -ne 0 ]; then
        OBSUTIL rm -f -r ${online_dict_dir}/${version} 1>>${log_path}
        exit 1
    fi

    __upload_cloud ${local_dict_dir}/assets.extra "${online_dict_dir}/${version}/" ${log_path}

    if [ $? -ne 0 ]; then
        OBSUTIL rm -f -r ${online_dict_dir}/${version} 1>>${log_path}
        exit 1
    fi

    __upload_cloud ${dict_txt} "${online_dict_dir}/${version}/" ${log_path}

    if [ $? -ne 0 ]; then
        OBSUTIL rm -f -r ${online_dict_dir}/${version} 1>>${log_path}
        exit 1
    fi

    __upload_cloud ${local_dict_dir}/saved_model.pb "${online_dict_dir}/${version}/" ${log_path}

    if [ $? -ne 0 ]; then
        OBSUTIL rm -f -r ${online_dict_dir}/${version} 1>>${log_path}
        exit 1
    fi

    __upload_cloud ${dict_md5} "${online_dict_dir}/${version}/" ${log_path}

    if [ $? -ne 0 ]; then
        OBSUTIL rm -f -r ${online_dict_dir}/${version} 1>>${log_path}
        exit 1
    fi

    NOTICE_LOG "Finished pushing model to HuaweiCloud" ${log_path}

    last_ver_tstamp=0
    last_ver_string=""
    all_vers=`${OBSUTIL} ls -s -d ${online_dict_dir} | grep "${ONLINE_DICT_NAME}"`
    for item in ${all_vers}
    do
        curr_ver_string=`basename ${item} | grep -E "[0-9]{14}"`
        curr_ver_tstamp=`echo ${curr_ver_string} | sed 's/\([0-9]\{8\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1 \2:\3:/'`
        curr_ver_tstamp=`date -d "${curr_ver_tstamp}" +%s`
        if [ "${curr_ver_string}" == "" -o "${curr_ver_string}" == "${version}" ]; then
            continue
        fi
        if [ $(FLOAT_GT ${curr_ver_tstamp} ${last_ver_tstamp}) -ne 0 ]; then
            last_ver_tstamp=${curr_ver_tstamp}
            last_ver_string=${curr_ver_string}
        fi
    done
    for item in ${all_vers}
    do
        curr_ver=`basename ${item} | grep -E "[0-9]{14}"`
        if [ "${curr_ver}" != "${version}" -a "${curr_ver}" != "${last_ver_string}" -a "${curr_ver}" != "" ]; then
            ${OBSUTIL} rm -f -r ${online_dict_dir}/${curr_ver}
            NOTICE_LOG "Delete online_model=${online_dict_dir}/${curr_ver}" ${log_path}
        fi
    done
}

function collect_model()
{
    version=$1  # train timestamp
    data_dir=$2
    log_path=$3

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
        exit 1
    fi

    # copy model to online dict dir
    dict_txt="${local_dict_dir}/${ONLINE_DICT_NAME}.txt"
    dict_md5="${local_dict_dir}/${ONLINE_DICT_NAME}.md5"

    while read line
    do
        info=(`echo "${line}" | jq --raw-output --exit-status \
            '.feature_name,.feature_type,.hash_threshold,.sequence_length,.emb_name'`)
        if [ $? -ne 0 ]; then
            ERROR_LOG "Parse feature schema line=${line} failed"
            exit 1
        fi
        fname="${info[0]}"
        ftype="${info[1]}"
        fsize="${info[2]}"
        flen="${info[3]}"
        femb="${info[4]}"
        if [ "${femb}" == "" ]; then
            WARNING_LOG "feature_name=${fname}, emb_name is empty" ${log_path}
            continue
        fi
        echo -e "${fname}\t${ftype}\t${fsize}\t${flen}\t${femb}" >> ${dict_txt}
    done < ${RUN_DIR}/data/${FEATURE_SCHEMA_FILE}

    if [ ! -f ${dict_txt} ]; then
        ERROR_LOG "Generate ${dict_txt} failed" ${log_path}
        exit 1
    fi

    echo -e "dnn_version\t${version}" >> ${dict_txt}

    md5sum ${dict_txt} | awk '{print $1}' >> ${dict_md5}

    NOTICE_LOG "Finished collecting model of ${version}" ${log_path}
}

function prepare_train()
{
    tstamp=$1
    data_dir=$2
    work_dir=$3
    log_path=$4

    NOTICE_LOG "Starting generating feature map" ${log_path}

    feature_lst_path="${work_dir}/feature_list.csv"
    feat_schema_file="${RUN_DIR}/data/${FEATURE_SCHEMA_FILE}"

    rm -f ${feature_lst_path}
    
    while read line
    do
        info=(`echo "${line}" | jq --raw-output --exit-status \
            '.feature_name,.feature_type,.hash_threshold,.sequence_length,.emb_name'`)
        if [ $? -ne 0 ]; then
            ERROR_LOG "Parse feature schema line=${line} failed"
            exit 1
        fi
        fname="${info[0]}"
        ftype="${info[1]}"
        fsize="${info[2]}"
        flen="${info[3]}"
        femb="${info[4]}"
        if [ "${femb}" == "" ]; then
            WARNING_LOG "feature_name=${fname}, emb_name is empty" ${log_path}
            continue
        fi
        echo -e "${fname},${ftype},${fsize},${flen},${femb}" >> ${feature_lst_path}
    done < ${RUN_DIR}/data/${FEATURE_SCHEMA_FILE}

    NOTICE_LOG "Finished generating feature map" ${log_path}

    if [ ! -d ${work_dir}/cfg ]; then
        mkdir -p ${work_dir}/cfg
    fi

    NOTICE_LOG "Starting generating training cfg" ${log_path}

    train_data_list="${work_dir}/cfg/train"
    valid_data_list="${work_dir}/cfg/valid"

    rm -f ${train_data_list} ${valid_data_list}

    for item in `ls ${data_dir}/${tstamp} | grep "^part"`
    do
        echo "${data_dir}/${tstamp}/${item}" >> ${train_data_list}
        echo "${data_dir}/${tstamp}/${item}" >> ${valid_data_list}
    done

    trn_list_size=`wc -l ${train_data_list} | awk '{print $1}'`
    val_list_size=`wc -l ${valid_data_list} | awk '{print $1}'`

    if [ ${trn_list_size} -eq 0 -o ${val_list_size} -eq 0 ]; then
        ERROR_LOG "${train_data_list} or ${valid_data_list} is null" ${log_path}
        exit 1
    fi
    cp ${valid_data_list} ${work_dir}/valid_data_list

    NOTICE_LOG "Finished generating training cfg" ${log_path}
}

function train()
{
    work_dir=$1
    log_path=$2

    NOTICE_LOG "Starting training model" ${log_path}
    train_model ${work_dir} ${log_path}
    NOTICE_LOG "Finished training model" ${log_path}
}

function eval()
{
    work_dir=$1
    log_path=$2
    feature_name=$3

    pred_path="${work_dir}/pred.txt"
    model_dir="${work_dir}/summaries/${MODEL}_${TASK_NAME}"

    valid_data_list="${work_dir}/valid_data_list"
    feature_list_path="${work_dir}/feature_list.csv"

    if [ ! -d ${model_dir} ]; then
        ERROR_LOG "Cannot find model_dir=${model_dir}" ${log_path}
        exit 1
    fi
    if [ ! -f ${valid_data_list} ]; then
        ERROR_LOG "Cannot find valid_data_list=${valid_data_list}" ${log_path}
        exit 1
    fi
    if [ ! -f ${feature_list_path} ]; then
        ERROR_LOG "Cannot find feature_list_path=${feature_list_path}" ${log_path}
        exit 1
    fi

    predict_py_route="./main.py"

    if [ "$feature_name" == "" ]; then
        feature_name="None"
        NOTICE_LOG "Starting evaluating model with all features" ${log_path}
    else
        NOTICE_LOG "Starting evaluating model without feature=${feature_name}" ${log_path}
    fi

    python3 ${predict_py_route} \
        --mode test \
        --attention_cols ${ATTENTION_COLS} \
        --schema ${feature_list_path} \
        --feature_influence ${feature_name} \
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
        ${@:2} \
        >> ${log_path} 2>&1
}

function __influence_out() 
{
    log_path=$1
    influence_file=$2
    feature_name=$3

    log_info=`grep "Saving dict for global step" ${log_path} | \
        tail -n 1 | sed s/[[:space:]]//g | awk -F ':|,' '{for(i=1;i<=NF;i++){if(match($i, "=")){print $i}}}'`

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
    echo "${feature_name},${auc},${loss},${score}" >> ${influence_file}
}

function feature_influence()
{
    work_dir=$1
    log_path=$2

    influence_file="${work_dir}/influence.csv"
    rm -f ${influence_file}
    echo "name,auc,loss,score" >> ${influence_file}

    NOTICE_LOG "Starting evaluating features influence" ${log_path}
    eval ${work_dir} ${log_path}
    __influence_out ${log_path} ${influence_file} "base"

    feature_list_path="${work_dir}/feature_list.csv"
    features=`cat $feature_list_path | awk -F ',' '{print $1}'`
    for feat in ${features}
    do
        eval ${work_dir} ${log_path} ${feat}
        __influence_out ${log_path} ${influence_file} ${feat}
    done
    NOTICE_LOG "Finished evaluating features influence" ${log_path}
}

function predict()
{
    work_dir=$1
    log_path=$2
    out_flag=$3

    pred_path="${work_dir}/pred.txt"
    model_dir="${work_dir}/summaries/${MODEL}_${TASK_NAME}"

    valid_data_list="${work_dir}/valid_data_list"
    feature_list_path="${work_dir}/feature_list.csv"

    if [ ! -d ${model_dir} ]; then
        ERROR_LOG "Cannot find model_dir=${model_dir}" ${log_path}
        exit 1
    fi
    if [ ! -f ${valid_data_list} ]; then
        ERROR_LOG "Cannot find valid_data_list=${valid_data_list}" ${log_path}
        exit 1
    fi
    if [ ! -f ${feature_list_path} ]; then
        ERROR_LOG "Cannot find feature_list_path=${feature_lst_path}" ${log_path}
        exit 1
    fi

    predict_py_route="./main.py"

    NOTICE_LOG "Starting predicting with model=${model_dir}" ${log_path}

    python3 ${predict_py_route} \
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
        ${@:2} \
        >> ${log_path} 2>&1

    NOTICE_LOG "Finished predicting with model=${model_dir}" ${log_path}

    NOTICE_LOG "Starting calculating metrics" ${log_path}
    is_user_type=`echo ${PRED_OUTPUT_FORMAT} | grep "user_type"`
    if [ "${is_user_type}" != "" ]; then
        cat ${pred_path} | python3 ${RUN_DIR}/scripts/auc.py main_user_type ${PRED_OUTPUT_FORMAT} ${out_flag} >> ${log_path} 2>&1
    else
        cat ${pred_path} | python3 ${RUN_DIR}/scripts/auc.py main ${PRED_OUTPUT_FORMAT} ${out_flag} >> ${log_path} 2>&1
    fi
    NOTICE_LOG "Finished calculating metrics" ${log_path}
}

function __clean_one_dir()
{
    data_dir=$1
    tstamp=$2
    shelf_life=$3

    clean_items=`ls ${data_dir} | grep -E "[0-9]{14}"`
    for item in ${clean_items}
    do
        item_format=`echo ${item} | sed 's/\([0-9]\{8\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1 \2:\3:/'`
        item_tstamp=`date -d "${item_format}" +%s`
        if [ $(FLOAT_SUB ${tstamp} ${item_tstamp}) -ge ${shelf_life} ]; then
            rm -rf ${data_dir}/${item}
            NOTICE_LOG "Delete ${data_dir}/${item}" ${log_path}
        fi
    done
}

function clean_data()
{
    std_format=`echo $1 | sed 's/\([0-9]\{8\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1 \2:\3:/'`
    std_tstamp=`date -d "${std_format}" +%s`
    __clean_one_dir "${BASE_DATA_DIR}" "${std_tstamp}" "${SHELF_LIFE_TRAIN_DATA}"
    __clean_one_dir "${BASE_ONLINE_DICT_LOCAL_DIR}" "${std_tstamp}" "${SHELF_LIFE_MODEL_DATA}"
    __clean_one_dir "${WORK_DIR}" "${std_tstamp}" "${SHELF_LIFE_MODEL_DATA}"
}

function send_mail()
{
    ver_date=$1
    work_dir=$2
    log_path=$3

    NOTICE_LOG "Starting sending mail for model updating" ${log_path}
    if [ ! -f ${work_dir}/monitor_test.txt ]; then
        ERROR_LOG "Cannot find ${work_dir}/monitor_test.txt" ${log_path}
        exit 1
    fi
    subject="${MAIL_SUBJECT} ${ver_date}"
    msg="<p>Congratulations! Your model is updated successfully. ^o^</p>"
    msg="${msg}<p></p>Key indicators as follows<br>"
    monitor_msg=`awk -F ' ' '{print "----> "$2"<br>"}' ${work_dir}/monitor_test.txt`
    msg="${msg}${monitor_msg}"
    python3 ${RUN_DIR}/scripts/send_mail.py "${MAIL_ADDRESS}" "${subject}" "${msg}"
    NOTICE_LOG "Finished sending mail for model updating" ${log_path}
}

function check_data()
{
    tstamp=$1
    data_dir=$2
    log_path=$3
    while true
    do
        part_num=`ls ${data_dir}/${tstamp} | grep -c "^part"`
        succ_num=`ls ${data_dir}/${tstamp} | grep -c "^_SUCCESS"`
        if [ ${part_num} -lt ${succ_num} -o ${part_num} -eq 0 ]; then
            WARNING_LOG "${data_dir}/${tstamp} is not ready yet" ${log_path}
            sleep 60
        else
            break
        fi
    done
}

function check_config()
{
    log_path=$1

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
    if [ "$ACTIVATION" == '' ]; then
        ACTIVATION="relu"
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
    if [ "${PIPELINE_CLEAN_DATA}" == "" ]; then
        PIPELINE_CLEAN_DATA="False"
    fi
    if [ "${SHELF_LIFE_TRAIN_DATA}" == "" -o $(FLOAT_LT ${SHELF_LIFE_TRAIN_DATA}) -ne 0 ]; then
        ERROR_LOG "SHELF_LIFE_TRAIN_DATA must be more than 0" ${log_path}
        exit 1
    fi
    if [ "${SHELF_LIFE_MODEL_DATA}" == "" -o $(FLOAT_LT ${SHELF_LIFE_MODEL_DATA}) -ne 0 ]; then
        ERROR_LOG "SHELF_LIFE_MODEL_DATA must be more than 0" ${log_path}
        exit 1
    fi
}

function pipeline()
{
    tstamp=$1
    data_dir=$2
    work_dir=$3
    log_path=$4

    # check data
    check_data ${tstamp} ${data_dir} ${log_path}

    # prepara_train
    prepare_train ${tstamp} ${data_dir} ${work_dir} ${log_path}

    # train
    train ${work_dir} ${log_path}

    # backup model
    backup_increment_model ${work_dir} ${log_path} ${tstamp}

    # collect_model
    collect_model ${tstamp} ${data_dir} ${log_path}

    if [ "${DO_EVAL}" == 'True' ]; then
        eval ${work_dir} ${log_path}

        monitor_eval ${tstamp} ${log_path} ${work_dir} ${TRN_VALID_FLAG} "test"

        error=$?
        if [ ${error} -gt 0 ]; then
            ERROR_LOG "Model is illegal, error code is ${error}" ${log_path}
            exit 1
        fi
    else
        # predict
        predict ${work_dir} ${log_path} ${TRN_VALID_FLAG} 

        # monitor
        monitor ${tstamp} ${log_path} ${work_dir} ${TRN_VALID_FLAG} "test"

        error=$?
        if [ ${error} -gt 0 ]; then
            ERROR_LOG "Model is illegal, error code is ${error}" ${log_path}
            exit 1 
        fi
    fi

    # push_online
    push_model ${tstamp} ${log_path}

    # send mail
    send_mail ${tstamp} ${work_dir} ${log_path}

    if [ ${PIPELINE_CLEAN_DATA} == "True" ]; then
        # clean data
        clean_data ${tstamp}
    fi
}

if [ $# -lt 2 ]; then
    echo "Usage: $0 [conf_file] [task_type] [format_date]"
    echo "[format_date] is required, format is [%Y%m%d_%H%M%S]"
    echo "[task_type] options as following"
    echo "-----   <prepare_train>    --  Generate cfg/feature_map etc. for training."
    echo "-----   <train>            --  Train model."
    echo "-----   <eval>             --  Eval model."
    echo "-----   <backup_model>     --  Backup model for incremental trainning."
    echo "-----   <predict>          --  Predict data with the model of [date]."
    echo "-----   <feat_influence>   --  Eval feature influence of [date]."
    echo "-----   <collect>          --  Collect offline model for online use."
    echo "-----   <monitor>          --  Check model and upload monitor indicators."
    echo "-----   <monitor_eval>     --  Check model and upload monitor indicators."
    echo "-----   <send_mail>        --  Send mail for updating new model."
    echo "-----   <push_online>      --  Push model to online server."
    echo "-----   <clean_data>       --  Clean expired data."
    echo "-----   <pipeline>         --  Execute a series of operations to update model."
    echo -e "e.g. $0 conf/config.sh train 20201210_170000\n"
    exit 1
fi

conf_file=$1

if [ ! -f ${conf_file} ]; then
    echo "[ERROR] ${conf_file} does not exist"
    exit 1
fi

source ${conf_file}

export RUN_DIR="$(pwd)/work/weshare"

if [ ! -f ${RUN_DIR}/scripts/util.sh ]; then
    echo "[ERROR] ${RUN_DIR}/scripts/util.sh does not exist"
    exit 1
fi

source ${RUN_DIR}/scripts/util.sh

tstamp=$3
task_type=$2

work_dir="${WORK_DIR}/${tstamp}"
if [ ! -d ${work_dir} ]; then
    mkdir -p ${work_dir}
fi
log_path="${work_dir}/run.log"
NOTICE_LOG "work_dir=${work_dir}" ${log_path}

check_config ${log_path}

SHELF_LIFE_TRAIN_DATA=$(FLOAT_MUL "${SHELF_LIFE_TRAIN_DATA}" "60")
SHELF_LIFE_MODEL_DATA=$(FLOAT_MUL "${SHELF_LIFE_MODEL_DATA}" "60")

export OBSUTIL="obsutil"

export TRN_VALID_FLAG="Final-train-valid"
export TST_VALID_FLAG="Final-test-valid"

if [ "${task_type}" == "prepare_train" ]; then
    prepare_train ${tstamp} ${BASE_DATA_DIR} ${work_dir} ${log_path}

elif [ "${task_type}" == "train" ]; then
    train ${work_dir} ${log_path}

elif [ "${task_type}" == "eval" ]; then
    eval ${work_dir} ${log_path}

elif [ "${task_type}" == "predict" ]; then
    predict ${work_dir} ${log_path} ${TST_VALID_FLAG}

elif [ "${task_type}" == "feat_influence" ]; then
    feature_influence ${work_dir} ${log_path}

elif [ "${task_type}" == "collect" ]; then
    collect_model ${tstamp} ${BASE_DATA_DIR} ${log_path}

elif [ "${task_type}" == "backup_model" ]; then
    backup_increment_model ${work_dir} ${log_path} ${tstamp}

elif [ "${task_type}" == "push_online" ]; then
    push_model ${tstamp} ${log_path}

elif [ "${task_type}" == "send_mail" ]; then
    send_mail ${tstamp} ${work_dir} ${log_path}

elif [ "${task_type}" == "monitor" ]; then
    monitor ${tstamp} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"

elif [ "${task_type}" == "monitor_eval" ]; then
    monitor_eval ${tstamp} ${log_path} ${work_dir} ${TST_VALID_FLAG} "test"

elif [ "${task_type}" == "clean_data" ]; then
    clean_data ${tstamp}

elif [ "${task_type}" == "pipeline" ]; then
    pipeline ${tstamp} ${BASE_DATA_DIR} ${work_dir} ${log_path}
     
else
    ERROR_LOG "Illegal task_type=${task_type}" ${log_path}
fi

exit 0
