#!/bin/bash

# Clean results folder
rm -f results/*

# Run the analysis on the simulation
echo "Running data-driven SIR model"
python3 main.py -s -mt 10

echo "Running data-driven SIR model with GT aggregation"
python3 main.py -s -gt -mt 10

echo "Running null SIR model"
python3 main.py -s -n -mt 10

echo "Running null SIR model with GT aggregation"
python3 main.py -s -n -gt -mt 10

# Run the analysis on the data
echo "Estimating TE from data"
for w in {2..6}
do
	echo "Wave $w"
	python3 main.py -w $w -mt 10

	echo "Wave $w GT"
	python3 main.py -w $w -gt -mt 10
done
