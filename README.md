# Namsel OCR
An OCR application focused on machine-print Tibetan text

DOCUMENTATION COMING SOON. 

Tested only on Ubuntu 14.04 and higher. 

An overview of the Namsel project can be found in our article in the journal Himalayan Linguistics: https://escholarship.org/uc/item/6d5781k5

Check out our library partner for already OCR'd digital text: http://tbrc.org. 

Get started:
$ bash ubuntu_install.sh

This will install required packages, build the cython modules, unpack datasets, and initiate training for the classifiers. Note that training takes (classify.py) takes up to an hour or more to complete.

For command options:
$ python namsel.py --help 
