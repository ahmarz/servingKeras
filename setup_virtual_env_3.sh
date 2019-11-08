#!/bin/bash

#sudo easy_install pip
#sudo pip install virtualenv

virtualenv -p python3 ~/servingKeras3
source ~/servingKeras3/bin/activate
pip install --upgrade pip
pip install setuptools -U
pip install wheel --upgrade
pip install -r requirements_3.txt
jupyter nbextensions_configurator enable --user
jupyter contrib nbextension install --user