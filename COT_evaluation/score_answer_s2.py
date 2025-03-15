import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict


import sys
# OpenAI
import openai


from utilities import *
# 
from models import gpt, infimm, qwen2

from prompts import demo_prompt_score


# load demo prompt
def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, inst, it):
    demo_prompt = demo_prompt.strip()
    full_prompt = demo_prompt.format(question = inst['query_' + str(it)], gt=inst['answer'], extraction=inst['extraction_step' + str(it)])
    return full_prompt


def match_answer(inst, api_key, model, it, quick_match=False):
    # quick match
    if quick_match:
        return '1' if inst['answer'] == inst['extraction'] else '0'
    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt_score, inst, it)
        extraction = model.get_response(image_path = None, user_prompt=full_prompt)
        return extraction.replace("Judgement:", "").strip()
    except Exception as e:
        print(e)
        print(f"Error in matching answer")

    return ""


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = ' '.join(response.split(' ')[-trunk_length:])
        return return_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--output_dir", type=str, default="./cot_results/final_2000")
    parser.add_argument("--output_file", type=str, default="deepseek_v3_extract.json")
    parser.add_argument('--save_file', type=str, default='deepseek_v3_score.json')
    parser.add_argument("--mode", type=str, default="Text_Centric")
    # match
    parser.add_argument('--quick_match', action='store_true', help='use rules to match answer for some problems')
    # output
    parser.add_argument('--save_every', type=int, default=1, help='save every n problems')
    parser.add_argument('--cache', action='store_true', help='cache results')
    parser.add_argument('--trunk_response', type=int, default=30, help='trunk response to the last n words')
    parser.add_argument('--api_key', type=str, help='api key for openai')
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2_vl",
        help="llm engine",
        choices=["gpt-3.5-turbo", "claude-2", "gpt4", "gpt-4-0613", "bard"],
    )
    # args
    args = parser.parse_args()

    # set api key

    answer_extraction_file =  args.output_dir + '/' + args.output_file.replace('.json','') + "_" + args.mode + ".json"
    result_file = answer_extraction_file.replace('extract', 'score')
    print(f"Reading {answer_extraction_file}...")
    rq = read_json(answer_extraction_file)
    results = {item['id']: item for item in rq}

    save_file =  args.output_dir + '/' + args.save_file.replace('.json','')  + "_" + args.mode + ".json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    save_results = []
    score_dict = defaultdict(list)
    temp_results = {item['id']: item for item in save_results}
    # read results
    if args.model == "qwen2_vl":
        model = qwen2.Qwen2VLInference()
    elif args.model == "infimm":
        model = infimm.InfimmInference()
    elif args.model == "gpt-4-turbo":
        model = gpt.GPT_Model(args.model, '1', patience=5)
    full_pids = list(results.keys())

    # tqdm, enumerate results
    for i, pid in enumerate(tqdm(full_pids)):
        save_inst = results[pid]
        for it, step_text in enumerate(save_inst["solution"]):
            if pid in temp_results:
                if temp_results[pid][f"score_step{it}"] != "None":
                    print(f"Results already exist for {pid} _ {it}.")
                    results[pid]["score_step"+ str(it)] = temp_results[pid]["score_step"+ str(it)]
                    continue
                
            judgement = match_answer(save_inst, args.api_key, model, it, args.quick_match)
            while True:
                if judgement.strip() not in ['0', '1']:
                    print('Wrong return format: ', judgement)
                    judgement = match_answer(save_inst, args.api_key, model, it, args.quick_match)
                else:
                    save_inst['score_step'+ str(it)] = int(judgement)
                    break
            save_results.append(save_inst)


        # score_dict[save_inst['question_type']].append(save_inst['judgement'])
        # score_version_dict[save_inst['problem_version']].append(save_inst['judgement'])

        if i % args.save_every == 0 or i == len(results)-1:
            print(f"Saving results to {result_file}...")
            save_json(save_results, result_file)
            print(f"Results saved.")
    
    # subject level acc
    # acc_result = {}
    # total_cnt, right_cnt = 0, 0
    # for subject in score_dict:
    #     subject_total_cnt = len(score_dict[subject])
    #     subject_right_cnt = len([inst for inst in score_dict[subject] if inst == 1])
    #     total_cnt += subject_total_cnt
    #     right_cnt += subject_right_cnt
    #     acc_result[subject] = {
    #         "Acc": (subject_right_cnt/subject_total_cnt),
    #         "Total": subject_total_cnt
    #     }
    # acc_result['average']= {
    #     "Acc": (right_cnt/total_cnt),
    #     "Total": total_cnt
    # }

    # save_json(acc_result, save_file)
    
