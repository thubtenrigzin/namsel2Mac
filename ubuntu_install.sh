#! /usr/bin/env bash
# Do all of this and then save a snapshot
sudo apt-get update
sudo apt-get install python-setuptools python-dev build-essential 
sudo apt-get install python-opencv libtiff-tools python-gtk2 python-cairo unzip
sudo apt-get install scantailor
sudo pip install scipy pillow sklearn cython requests simplejson
python setup.py build_ext --inplace
mkdir -p ~/.fontscp data_generation/fonts/*ttf ~/.fonts/
fc-cache -f -v
cd data_generation
python font_draw.py
cd ..
cd datasets
unzip datapickles.zip
cd ..
python classify.py