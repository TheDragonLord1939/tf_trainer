#!/bin/bash

function ERROR_LOG()
{
    log_time=`date "+%Y-%m-%d %H:%M:%S"`
    echo -e "\033[31;1m[${log_time}] [ERROR] $1\033[0m"
    if [ $# -eq 2 ]; then
        echo -e "[${log_time}] [ERROR] $1" >> $2
    fi
}

function WARNING_LOG()
{
    log_time=`date "+%Y-%m-%d %H:%M:%S"`
    echo -e "\033[33;1m[${log_time}] [WARNING] $1\033[0m"
    if [ $# -eq 2 ]; then
        echo -e "[${log_time}] [WARNING] $1" >> $2
    fi
}

function NOTICE_LOG()
{
    log_time=`date "+%Y-%m-%d %H:%M:%S"`
    echo -e "\033[32;1m[${log_time}] [NOTICE] $1\033[0m"
    if [ $# -eq 2 ]; then
        echo -e "[${log_time}] [NOTICE] $1" >> $2
    fi
}

function FLOAT_MUL()
{
    echo "$1 $2" | awk '{print $1 * $2}'
}

function FLOAT_SUB()
{
    echo "$1 $2" | awk '{print $1 - $2}'
}

function FLOAT_ADD()
{
    echo "$1 $2" | awk '{print $1 + $2}'
}

function FLOAT_LT()
{
    echo "$1 $2" | awk '{if($1 < $2){print 1}else{print 0}}'
}

function FLOAT_GT()
{
    echo "$1 $2" | awk '{if($1 > $2){print 1}else{print 0}}'
}
