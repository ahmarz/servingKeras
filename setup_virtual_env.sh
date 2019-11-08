#!/bin/bash

#sudo easy_install pip
#sudo pip install virtualenv

virtualenv -p python ~/servingKeras
source ~/servingKeras/bin/activate
pip install --upgrade pip
pip install setuptools -U
pip install -r requirements.txt
jupyter nbextensions_configurator enable --user
jupyter contrib nbextension install --user