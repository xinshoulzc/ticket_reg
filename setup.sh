#!/bin/sh

# Check Python version
echo "check python3 version, Waiting..."

PY_VERSION=`python3 -V 2>&1|awk '{print $2}'`
if [ $PY_VERSION \> "3.7.5" ]
then
  echo "python version ok!"
else
  echo "check your python version >= 3.7.6, Now is python $PY_VERSION"
  exit -1
fi

# Check pip version
echo "check pip3 version, Waiting..."

PIP_VERSION=`pip3 -V 2>&1|awk '{print $2}'`
if [ $PIP_VERSION \> "19.3.0" ]
then
  echo "python version ok!"
else
  echo "check your pip3 version >= 19.3.1, Now is pip $PIP_VERSION"
  exit -1
fi

# create log dir
if [ ! -d "log" ]
then
  echo "create log dir"
  mkdir log
fi

# create dataset dir
if [ ! -d "dataset/raw" ]
then
  echo "create dataset dir"
  mkdir -p dataset/raw
fi

install requirement package
echo "install requirement package, Waiting..."
echo

pip3 install -r requirements.txt
