#! /bin/bash
cd $HOME/letters/
kernprof.py -l recognize.py
python -m line_profiler recognize.py.lprof 
