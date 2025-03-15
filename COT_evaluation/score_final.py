
import os
import copy
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import sys
# OpenAI
import openai
from utilities import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    # parser.add_argument("--data_dir", type=str, default="./cot_results/final_606")
    parser.add_argument("--data_dir", type=str, default="./cot_results/final_2000")
    parser.add_argument('--input_file', type=str, default='deepseek_v3_score.json')
    parser.add_argument('--save_file', type=str, default='deepseek_v3_final_score.json')
    parser.add_argument("--mode", type=str, default="Text_Centric")
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    # args
    args = parser.parse_args()

    # read results
    input_file =  args.data_dir + '/' + args.input_file.replace('.json','') + "_" + args.mode + ".json"
    save_file =  args.data_dir + '/' + args.save_file.replace('.json','')  + "_" + args.mode + ".json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    score_dict = defaultdict(lambda: defaultdict(list))
    rq = read_json(input_file)
    results = {item['id']: item for item in rq}

    for k, inst in results.items():
        final_score = 0
        for i in range(len(inst['solution'])):
            score_dict[inst['question_type']]['step_' + str(i)].append(inst['score_step' + str(i)])
            final_score += inst['score_step' + str(i)]
        final_score /= len(inst['solution'])
        score_dict[inst['question_type']]['final_score'].append(final_score)
    # subject level acc
    acc_result = {}
    total_cnt, right_score = 0, 0
    step_num = defaultdict(list)
    step_scoreS = defaultdict()
    for subject in score_dict:
        acc_result[subject] = {}
        for step in score_dict[subject]:
            acc_result['average'] = {step: 0}
            if step == 'final_score':
                step_total_cnt = len(score_dict[subject][step])
                total_cnt += step_total_cnt
                step_score = np.sum(np.array(score_dict[subject][step]))
                right_score += step_score

                acc_result[subject]["final_score"] = (step_score/step_total_cnt)
                acc_result[subject]["Total"] = step_total_cnt
            else:
                acc_result[subject][step] = np.mean(np.array(score_dict[subject][step]))
                step_num[step].append(subject)

    for subject in acc_result:
        for step in acc_result[subject]:
            if step == 'final_score' or step == 'Total':
                continue

            step_scoreS[step] = step_scoreS.get(step, 0) + acc_result[subject][step]
            

    for step in step_num.keys():
        step_scoreS[step] /= len(step_num[step])
        acc_result['average'][step] = step_scoreS[step]
        
    acc_result['average']["Acc"]= (right_score/total_cnt)
    acc_result['average']["Total"]= total_cnt

    
    save_json(acc_result, save_file)
