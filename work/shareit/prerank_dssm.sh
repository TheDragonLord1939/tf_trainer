#!/bin/bash
export TZ='Asia/Shanghai'
. ../commont_utils/common.sh

conf_file="conf/prerank_dssm_conf.sh"

echo "========== BEGIN `date` =========="
i=1
dt=`date -d "$i days ago" +%Y%m%d`
yes_dt=`date -d "$dt -1 days" +%Y%m%d`

# 将昨天的模型结果变成增量训练目录
bash run.sh $conf_file "backup" $yes_dt

bash run.sh $conf_file "download" $dt
if [ $? -ne 0 ]; then
    send_ta_alert "[Prerank] DSSM get_train_data $dt fail!" "18612981582"
    exit
fi


bash run.sh $conf_file "train" $dt
if [ $? -ne 0 ]; then
    send_ta_alert "[Prerank] DSSM train_model $dt fail!" "18612981582"
    exit
fi


bash run.sh $conf_file "push_model" $dt
if [ $? -ne 0 ]; then
    send_ta_alert "[Prerank] DSSM push_model $dt fail!" "18612981582"
    exit
fi

echo "========== FINISH `date` =========="
