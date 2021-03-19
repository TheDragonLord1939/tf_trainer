#!/bin/bash
export TZ='Asia/Shanghai'
. ../commont_utils/common.sh

conf_file="conf/recall_dssm_conf.sh"

echo "========== BEGIN `date` =========="
dt=`date -d "1 days ago" +%Y%m%d`
if [ $# -eq 1 ]; then
    dt=$1
fi
yes_dt=`date -d "$dt -1 days" +%Y%m%d`
clean_data_dt=`date -d "$dt -3 days" +%Y%m%d`

receviers="xxx@ushareit.com"
subject="[TF_TRAINER]_RECALL_DSSM"

# 将昨天的模型结果变成增量训练目录
bash run.sh $conf_file "backup" $yes_dt

bash run.sh $conf_file "download" $dt
if [ $? -ne 0 ]; then
     python3 ../commont_utils/send_mail.py $receviers $subject "get_train_data $dt fail!"
     exit
fi


bash run.sh $conf_file "train" $dt
if [ $? -ne 0 ]; then
    python3 ../commont_utils/send_mail.py $receviers $subject "train $dt fail!"
    exit
fi

bash run.sh $conf_file "eval" $dt
if [ $? -ne 0 ]; then
	python3 ../commont_utils/send_mail.py $receviers $subject "eval $dt fail!"
	exit
else
	log=`grep loss "/tf/tensorflow/tf_trainer/work/shareit/logs/dssm_eval_${dt}.log"`
    python3 ../commont_utils/send_mail.py $receviers $subject "$log"
fi

bash run.sh $conf_file "predict" $dt
if [ $? -ne 0 ]; then
    python3 ../commont_utils/send_mail.py $receviers $subject "predict $dt fail!"
    exit
fi

bash run.sh $conf_file "export_new" $dt
if [ $? -ne 0 ]; then
    python3 ../commont_utils/send_mail.py $receviers $subject "export_new $dt fail!"
    exit
fi

bash run.sh $conf_file "save_item_embedding_new" $dt
if [ $? -ne 0 ]; then
    python3 ../commont_utils/send_mail.py $receviers $subject "save_item_embedding_new $dt fail!"
    exit
fi

bash run.sh $conf_file "push_model" $dt
if [ $? -ne 0 ]; then
    python3 ../commont_utils/send_mail.py $receviers $subject "push_model $dt fail!"
    exit
fi

bash run.sh $conf_file "monitor" $dt
if [ $? -ne 0 ]; then
    python3 ../commont_utils/send_mail.py $receviers $subject "monitor $dt fail!"
    exit
fi

# bash run.sh $conf_file "feat_influence" $dt $receviers
# if [ $? -ne 0 ]; then
#     python3 ../commont_utils/send_mail.py $receviers $subject "feat_influence $dt fail!"
#     exit
# fi

bash run.sh $conf_file "clean_data" $clean_data_dt

echo "========== FINISH `date` =========="
