import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
import sys

from models import claude, gpt, bard, infimm, qwen2
# from utils import *

# OpenAI
import openai
from utilities import *
# os.environ["OVERRIDE_HEADER"] = "1"  # !!!!必须要加这个，否则会报错
from prompts import demo_prompt_extract


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt


def extract_answer(response, model):    
    if len(response) < 50: # 字数太少，不用提取 
        return response
    # general extraction

    try:
        full_prompt = create_test_prompt(demo_prompt_extract, response,)
        extraction = model.get_response(image_path = None, user_prompt=full_prompt)
        # extraction = get_chat_response(full_prompt, api_key)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {response}")
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
    parser.add_argument("--output_file", type=str, default="deepseek_v3.json")
    parser.add_argument("--mode", type=str, default="Text_Centric")
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
    if args.model == "qwen2_vl":
        model = qwen2.Qwen2VLInference()
    elif args.model == "gpt-4-turbo":
        model = gpt.GPT_Model(args.model, '1', patience=5)
    output_file = args.output_file.replace('.json','') + "_" + args.mode + ".json"
    result_file = os.path.join(args.output_dir, output_file) # 读取的文件
    print(f"Reading {result_file}...")
    results = read_json(result_file)
    save_file = os.path.join(args.output_dir, args.output_file.replace('.json',f'_extract_{args.mode}.json')) # 保存的文件

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if os.path.exists(save_file):
        save_results = json.load(open(save_file))
    else:
        save_results = []

    score_dict = defaultdict(lambda: defaultdict(list))
    score_dict_record = defaultdict(list)
    score_version_dict = defaultdict(list)

    full_pids = list(results.keys())
    test_num = len(full_pids)
    print("Number of problems to run:", test_num)
    temp_results = {item['id']: item for item in save_results}

    # enumerate results
    for i, pid in enumerate(tqdm(full_pids)):
        # if pid in save_results:
        #     print(f"Results already exist for {pid}.")
        #     continue
        save_inst = results[pid]
        # save_inst = save_results[i] if i < len(save_results) else copy.deepcopy(inst)
        if args.cache and 'extraction' in save_inst:
            pass
        else:
            for it, step_text in enumerate(save_inst["solution"]):
                if pid in temp_results:
                    if temp_results[pid][f"extraction_step{it}"] != "None":
                        print(f"Results already exist for {pid} _ {it}.")
                        results[pid]["extraction_step"+ str(it)] = temp_results[pid]["extraction_step"+ str(it)]
                        continue
                # if 'response' in save_inst:
                #     response = save_inst['response']
                # else:
                #     response = ''
                #     print(save_inst)
                #     print("######### NO MODEL ANSWER ###########")  # some model may output nothing due to safety
                print(f"\nextracting answer for {pid} _ {it}...")
                response = trunk_response(results[pid]['response_step'+ str(it)], args.trunk_response) # 去除response中的一些无用信息
                extraction  = extract_answer(response, model)
                if  isinstance(save_inst['choices'], list) and save_inst['choices'][0] != 'None': # 是选择题
                    if extraction == 'A':
                        extraction = save_inst['choices'][0]
                    elif extraction == 'B':
                        extraction = save_inst['choices'][1]
                    elif extraction == 'C':
                        extraction = save_inst['choices'][2]
                    elif extraction == 'D':
                        extraction = save_inst['choices'][3]
                    # elif extraction == 'E':
                    #     extraction = save_inst['choices'][4]
                save_inst['extraction_step'+ str(it)] = extraction.replace('Extracted Answer: ', '').strip()  # sometimes gpt will repeat
        save_results.append(save_inst)

        if i % args.save_every == 0 or i == len(results) - 1:
            print(f"Saving results to {save_file}...")
            save_json(save_results, save_file)
            print(f"Results saved.")