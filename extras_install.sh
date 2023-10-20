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

os_name=$(grep -o '^NAME=.*' /etc/os-release | cut -f2 -d\" | sed 's/"//g')
os_version=$(grep -o '^VERSION_ID=.*' /etc/os-release | cut -f2 -d\" | sed 's/"//g')
echo "Platform is ${os_name}, Version: ${os_version}"

ID=$(cat /etc/os-release | grep -w ID | cut -f2 -d"=")

# Requirements are from this link
# https://github.com/google-coral/examples-camera/blob/master/opencv/install_requirements.sh

if [ ${ID} = "raspbian" ]; then
   python3 -m pip install opencv-contrib-python==4.1.0.25
fi

if [ ${ID} = "ubuntu" ]; then
   python3 -m pip install --upgrade pip
   python3 -m pip install opencv-contrib-python==4.6.0.66
fi

if [ ${ID} = "mendel" ]; then

   git clone https://github.com/pjalusic/opencv4.1.1-for-google-coral.git /tmp/opencv_coral
   cp /tmp/opencv_coral/cv2.so /usr/local/lib/python3.7/dist-packages/cv2.so
   sudo cp -r /tmp/opencv_coral/libraries/. /usr/local/lib
   rm -rf /tmp/opencv_coral

fi

if [ "${os_name}" = "Ubuntu" ] && [ "${os_version}" = "20.04" ]; then
   # For ubuntu 20.04 install tflite runtime == 2.8.0
   python3 -m pip install tflite_runtime==2.8.0
else

   # For every other platform install tflite runtime by constraucting a url.
   py=$(python3 -V | awk '{print $2}' | awk -F. '{print $1 $2}')
   # Get the python version for this platform
   arch=$(uname -m)
   # Get the architecture for this platform.
   url=$(echo -n "https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp"; echo -n $py; echo -n "-cp"; echo -n $py; echo -n "m-linux_"; echo -n ${arch}; echo -n ".whl")
   echo "Going to fetch pip package for tflite runtime from $url" 
   python3 -m pip install $url
fi

if [ ${ID} != "mendel" ]; then
  echo "In order to use Edge TPU, please install edge TPU runtime, libedgetpu1-std
https://coral.ai/software/#debian-packages
note: This is pre-installed on Coral Dev Board."
fi
