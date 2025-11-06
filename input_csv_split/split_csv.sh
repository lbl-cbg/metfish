#!/bin/bash
# Usage: ./split_csv.sh input.csv output_dir

input_csv="$1"
output_dir="$2"

mkdir -p "$output_dir"

# Read header
header=$(head -n 1 "$input_csv")

tail -n +2 "$input_csv" | while IFS=, read -r name seqres; do
    name_clean=$(echo "$name" | tr -d '\r\n ')
    echo "$header" > "$output_dir/${name_clean}.csv"
    echo "$name,$seqres" >> "$output_dir/${name_clean}.csv"
done
