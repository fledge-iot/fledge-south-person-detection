#!/usr/bin/env bash

#    Copyright (c) 2020 Dianomic Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

##
## Author: Amandeep Singh Arora , Deepanshu Yadav
##

# The Rpi -4 requirements are from https://pimylifeup.com/raspberry-pi-opencv/
set -e

ID=$(cat /etc/os-release | grep -w ID | cut -f2 -d"=")
if [[ ${ID} == "raspbian" ]]; then
    # For handling images
    sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng-dev

    # For handling videos
    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libopenexr-dev
    sudo apt-get install libxvidcore-dev libx264-dev

    # have removed libgdk-pixbuf2.0-dev libpango1.0-dev libfontconfig1-dev libcairo2-dev python3-pyqt5 not required by plugin
    # For Displaying guis
    sudo apt-get install libgtk2.0-dev libgtk-3-dev

    # For speeding things up
    sudo apt-get install libatlas-base-dev gfortran

    # For hdf5 datasets and QT compatibility
    # These are required if we dont compile opencv.
    sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-103
    sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test

else
    echo "Installing Opencv "
    sudo apt-get install python3-opencv
fi

./extras_install.sh

sudo apt-get install v4l-utils