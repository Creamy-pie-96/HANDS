#!/bin/bash

input_dir="All stickers in one image/New"
output_dir="All stickers in one image/New/output"

mkdir -p "$output_dir"

for img in "$input_dir"/*; do
    filename=$(basename "$img")
    convert "$img.png" -fuzz 10% -transparent white "$output_dir/$filename"
done
