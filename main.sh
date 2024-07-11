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

echo "Found input file: $input_file"

# Define the paths to the Python scripts
premodel_script="preModel.py" 
model_script="model.py"
postmodel_script="postModel.py"

# Execute the preModel script with the input file
echo "Running preModel..."
output_from_premodel_script=$(python $premodel_script $input_file | tail -n 1)
echo "Output from preModel: $output_from_premodel_script"

# #Execute Model
echo "Running model..."
output_from_model_script=$(python $model_script "inference" "$output_from_premodel_script")
echo "Output from Model: $output_from_model_script"

# # Similarly, assuming 'model.py' outputs something 'postModel.py' can pick up
# # For example, 'model_output.mid' that postModel uses
# # echo "Running postModel..."
# # python $postmodel_script

# echo "Processing complete."
