#!/bin/bash

inputs=("Text_Centric" "Text_Limited" "Vision_Dense" "Vision_Centric" "Vision_Primary" "Text_Plus") # "v1" 
for input in "${inputs[@]}"
do
  python ./COT_evaluation/score_answer_s2.py --mode "$input" 
done

wait