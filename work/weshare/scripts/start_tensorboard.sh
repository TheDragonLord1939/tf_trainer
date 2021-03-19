#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 [tensorboard_port] [local_port]"
    exit 0
fi

tensorboard_port=$1
local_port=$2

docker_port="2214"

password="shareit"

expect -c "
set timeout 3600
spawn ssh -L ${local_port}:127.0.0.1:${tensorboard_port} root@192.168.5.68 -p ${docker_port}
expect {
    \"(yes/no)?\" { send \"yes\n\"; }
    \"*assword\" { send \"${password}\n\"; }
}
expect eof 
"
