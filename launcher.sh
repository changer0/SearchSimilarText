#!/bin/zsh
# 准备在这个地方编写启动脚本
curPath=`pwd`
scriptDir=`dirname $0`
if [ $scriptDir == "." ]
    then
    scriptDir=$curPath
fi
cd $scriptDir/src
pwd

python3 main.py