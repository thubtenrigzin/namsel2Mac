#!/bin/sh

#Homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

#Python 2.7
brew install python2

#Installing more stuffs
brew install pygtk
python2 -m pip install cython numpy pillow sklearn scipy simplejson opencv-contrib-python
#Downgrade scikit-learn version to 0.18.1
python2 -m pip install --upgrade scikit-learn==0.18.1

#Wget
brew install wget

#MacPort
wget https://github.com/macports/macports-base/releases/download/v2.4.2/MacPorts-2.4.2.tar.bz2
tar xzvf MacPorts-2.4.2.tar.bz2
rm MacPorts-2.4.2.tar.bz2
cd MacPorts-2.4.2
./configure && make && sudo make install
cd ..
rm -r MacPorts-2.4.2
export PATH="/opt/local/bin/:$PATH"
sudo port -v selfupdate

#Scantailor
sudo port install scantailor

#Here we are! Namsel OCR
python2 setup.py build_ext --inplace
mkdir -p ~/.fontscp ~/.fonts
fc-cache -f -v
cd data_generation
python2 font_draw.py
cd ..
cd datasets
unzip datapickles.zip
cd ..
python2 classify.py

echo -e '\nAll Done!\n\nRemember to add /opt/local/bin in your $PATH for Scantailor-cli to be called by Namsel before running the program (export PATH="/opt/local/bin/:$PATH")\n'
