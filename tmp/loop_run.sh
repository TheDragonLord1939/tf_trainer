#!/bin/bash


for((i=2;i>0;i--));do
day=$(date -d "$i day ago" +%Y%m%d)
echo $day

bash ./work/weshare/run.sh ./work/weshare/conf/config_nt_ad.sh pipeline1 $day

done

