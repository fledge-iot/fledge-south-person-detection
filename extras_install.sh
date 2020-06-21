#!/usr/bin/env bash

##--------------------------------------------------------------------
## Copyright (c) 2020 Dianomic Systems Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##--------------------------------------------------------------------

##
## Author: Ashish Jabble
##

ID=$(cat /etc/os-release | grep -w ID | cut -f2 -d"=")

# Requirements are from this link
# https://github.com/google-coral/examples-camera/blob/master/opencv/install_requirements.sh

if [ ${ID} = "raspbian" ]; then
   pip3 install opencv-contrib-python==4.1.0.25
fi

py=$(python3 -V | awk '{print $2}' | awk -F. '{print $1 $2}')
arch=$(uname -m)
url=$(echo -n "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp"; echo -n $py; echo -n "-cp"; echo -n $py; echo -n "m-linux_"; echo -n ${arch}; echo -n ".whl")
pip3 install $url

if [ ${ID} != "mendel" ]; then
  echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list;
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -;
  sudo apt-get update;
  sudo apt-get install libedgetpu1-std;
fi
