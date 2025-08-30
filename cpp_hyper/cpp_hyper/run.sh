#!/bin/bash

# Run the C++ program in current directory
g++ initial_flat.cpp -o initial_flat
./initial_flat


python3 /path/to/your/flat_opti.py

echo "Running simplex_hull.py..."
python3 simplex_hull.py

echo "All programs completed."
