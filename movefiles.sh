#!/bin/bash

# Define the source and destination folders
OUTPUT_FOLDER="output"
LIGHTCURVES_FOLDER="lightcurves"

# Loop through each subfolder in the output folder
for subfolder in $OUTPUT_FOLDER/*; do
    # Get the name of the subfolder
    subfolder_name=$(basename "$subfolder")

    # Create a corresponding subfolder in the lightcurves folder
    mkdir -p "$LIGHTCURVES_FOLDER/$subfolder_name"

    # Copy specific files ending in _lc.txt to the corresponding subfolder in lightcurves folder
     cp "$subfolder"/*_lc.txt "$LIGHTCURVES_FOLDER/$subfolder_name/"
     cp "$subfolder"/*_periodogram.txt "$LIGHTCURVES_FOLDER/$subfolder_name/"
done

