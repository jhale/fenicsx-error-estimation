#!/bin/bash

cd sines_bp_vs_bura/
echo "Run sines_bp_vs_bura"
python3 demo_bp_vs_bura.py
cd ../
echo "Plot sines_bp_vs_bura"
python3 plot.py
cd checkerboard_bp_vs_bura/
echo "Run checkerboard_bp_vs_bura"
python3 demo_bp_vs_bura.py
cd ../
echo "Plot checkerboard_bp_vs_bura"
python3 plot.py