#!/bin/bash

# Directory containing the input file
input_dir="Input/"

# Find the first file in the input directory (not in subdirectories)
input_file=$(find "$input_dir" -maxdepth 1 -type f | head -1)

# Check if the input directory contains exactly one file
file_count=$(find "$input_dir" -maxdepth 1 -type f | wc -l)
if [ "$file_count" -ne 1 ]; then
    echo "There must be exactly one file in the input directory."
    exit 1
fi

echo -e "Found input file: $input_file \n"

# Define the paths to the Python scripts
premodel_script="preModel.py" 
model_script="model.py"
postmodel_script="postModel.py"

# Get the optional temperature parameter
temperature=${1:-1.0}

# Execute preModel 
echo "Running preModel..."
output_from_premodel_script=$(python $premodel_script $input_file | tail -n 1)
echo -e "Output from preModel: $output_from_premodel_script \n"

# Execute Model
echo "Running model..."
output_from_model_script=$(python $model_script "inference" "$output_from_premodel_script")
#Splicing the [] from start and end of string
# output_from_model_script=${output_from_model_script:1:-1}
echo -e "Output from Model: $output_from_model_script \n"

# Execute PostModel
echo "Running model..."
python $model_script "inference" "$output_from_premodel_script" --temperature $temperature

echo -e "\n\nView Output MIDI: ChordGen\Input\Output\inference_song_midi.mid"
