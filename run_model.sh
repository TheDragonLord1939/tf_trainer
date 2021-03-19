#!/bin/bash

day=$(date -d "1 day ago" +%Y%m%d)

bash ./work/weshare/run.sh ./work/weshare/conf/config_nt_ad.sh pipeline1 $day
