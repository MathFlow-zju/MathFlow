#!/bin/bash

inputs=("Text_Centric" "Text_Limited" "Vision_Dense" "Vision_Centric" "Vision_Primary" "Text_Plus") # "v1"  "v7"
for input in "${inputs[@]}"
do
  python ./COT_evaluation/score_final.py --mode "$input" &
done

wait