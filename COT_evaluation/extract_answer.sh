#!/bin/bash

inputs=( "Text_Centric" "Text_Limited" "Vision_Dense" "Vision_Centric" "Vision_Primary" "Text_Plus") 
for input in "${inputs[@]}"
do
  python ./COT_evaluation/extract_answer_s1.py --mode "$input" 
done

wait