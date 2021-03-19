#!/usr/bin/bash
# 通用工具类脚本，支持如下功能：
# 1. 日志功能：
#	(1)log_info
#	(2)log_error
# 2. obs传输功能
# (1)查询数据是否存在check_file
#	(2)下载数据download_dir, download_file
# (3)上传数据upload_file
# (4)从obs上删除目录delete_file
# 3. 报警功能
#	(1)发送邮件send_mail
#	(2)发送钉钉报警send_alert

export TZ='Asia/Shanghai'
mkdir -p logs


log_info(){
    msg=$1
    log_file=$2

    log_time=`date "+%Y-%m-%d %H:%M:%S"`
    echo -e "${log_time} | INFO  | $msg"
    if [ $# -eq 2 ]; then
        echo -e "${log_time} | INFO  | $msg" >> $log_file
    fi
}


log_error(){
    msg=$1
    log_file=$2

    log_time=`date "+%Y-%m-%d %H:%M:%S"`
    echo -e "${log_time} | ERROR | $msg"
    if [ $# -eq 2 ]; then
        echo -e "${log_time} | ERROR | $msg" >> $log_file
    fi
}


download_data(){
    obs_dir=$1
    des_dir=$2

    # 日志
    log="logs/obsutil_download.log_`date +%s`"

    # 拉取数据
    obsutil cp -f -r -flat $obs_dir $des_dir > $log 2>&1
    
    # 失败重新拉取
    fail_num=`grep "Failed count is:" $log | awk '{print $NF}'`
    task_id=`grep "Task id is:" $log | awk '{print $NF}'` 
    while [[ x"" != x"$fail_num" ]] && [[ $fail_num -gt 0 ]];
    do
        log_error "$task_id fail $fail_num, retrying..."
        obsutil cp -f -r -flat -recover=$task_id > $log 2>&1
        fail_num=`grep "Failed count is:" $log | awk '{print $NF}'`
        task_id=`grep "Task id is:" $log | awk '{print $NF}'` 
    done
    
    rm -rf $log
}

# 上传数据
upload_data(){
    src_file=$1
    obs_dir=$2

    log="logs/obsutil_upload.log_`date +%s`"
    retry_num=0
    max_retry=10
    while [ $retry_num -lt $max_retry ]
    do
        obsutil cp -f -r -flat $src_file "${obs_dir}" >> $log 2>&1
        if [ $? -ne 0 ]; then
            log_error "upload $src_file to $obs_dir fail! retry $retry_num time(s)"
            retry_num=$((retry_num + 1))
        else
            rm -rf $log
            return 0                                            # 返回成功
        fi  
    done
    rm -rf $log
    return 1                                                    # 返回失败
}


# 下载目录
download_dir(){
    obs_dir=$1
    des_dir=$2

    begin_ts=`date +%s`
    # 确保目录为空
    rm -rf $des_dir && mkdir -p $des_dir
    download_data $obs_dir $des_dir
    end_ts=`date +%s`
    cost_ts=$(($end_ts - $begin_ts))
    log_info "[$obs_dir] ==> [$des_dir] cost $cost_ts s"
}


# 下载文件
download_file(){
    obs_dir=$1
    des_file=$2

    begin_ts=`date +%s`
    # 数据先下载到临时目录，然后再合并成文件
    tmp_dir="./tmp_`date +%s`"
    rm -rf $tmp_dir && mkdir -p $tmp_dir
    download_data $obs_dir $tmp_dir
    cat $tmp_dir/* > $des_file
    rm -rf $tmp_dir
    end_ts=`date +%s`
    cost_ts=$(($end_ts - $begin_ts))
    log_info "[$obs_dir] ==> [$des_file] cost $cost_ts s"
}


# 查询文件是否存在
check_file(){
    obs_file=$1					# 如果需要check的是目录，则参数一定要带/
    log="logs/obsutil_check.log_`date +%s`"
    obsutil stat $obs_file > $log 2>&1
    if [ $? -eq 0 ]; then
      rm -rf $log
      return 0
    else
      rm -rf $log
      return 1
    fi
}


# 上传文件
upload_file(){
    up_file=$1
    obs_dir=$2

    # 上传之前先check目录是否存在
    check_file $obs_dir
    if [ $? -eq 0 ]; then
      log_error "$obs_dir exist! Please delete it before upload!"
      return 1
    fi

    begin_ts=`date +%s`
    upload_data $up_file $obs_dir
    if [ $? -ne 0 ]; then
        log_error "[$up_file] ==> [$obs_dir] fail!"
        return 1
    fi

    # 创建标志位文件
    succ_file="_SUCCESS"
    touch $succ_file            # 若文件存在则修改文件的修改时间，若文件不存在则创建

    upload_data $succ_file $obs_dir
    end_ts=`date +%s`
    cost_ts=$(($end_ts - $begin_ts))
    log_info "[$up_file] ==> [$obs_dir] cost $cost_ts s"
    return 0
}


delete_file(){
    obs_dir=$1
    log="logs/obsutil_delete.log_`date +%s`"
    obsutil rm -f -r $obs_dir > $log 2>&1
    rm -rf $log
}


# 发送邮件
# 示例：send_mail "chengtx@ushareit.com" "test"
send_mail(){
    addrs=$1
	subject=$2
	msg=$3           # 可选
    python3 send_mail.py $addrs $subject $msg
}


# 发送到TA&新用户 core这个钉钉群，必须包含关键字[失败，fail，test，delay]
# 示例：send_alert "dnn模型更新失败" "18612981582"
send_ta_alert(){
    text=$1
    phone=$2

	# 多个收件人
	addrs=`echo $phone | awk -F',' '{for(i=1;i<NF;i++)printf("\""$i"\", ")}END{print "\""$NF"\""}'`

    msg='{"msgtype": "text", "text": {"content": "'$text'"}, "at": {"atMobiles": ['$addrs']}}'

    cmd="curl 'https://oapi.dingtalk.com/robot/send?access_token=7c8927d25bb451c221836d4f43facd38e53526660c48fbd7f12431a805948015' -H 'Content-Type: application/json' -d '$msg'"
    eval $cmd > /dev/null 2>&1
}

