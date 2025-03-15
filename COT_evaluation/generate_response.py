import os
# os.environ["OVERRIDE_HEADER"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

import io
import time
import argparse

from tqdm import tqdm

import sys

from utilities import *

from models import claude, gpt, bard, qwen2, llava_cot, internVL2_5_78, QVQ, Deepseek, Qwen2_5,llama3

from build_query import create_query_data, create_one_query_cot



def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--data_dir", type=str, default="./DATA") 
    parser.add_argument("--mode", type=str, default="v7") # ["v1", "v2", "v3", "v4", "v5", "v6"]
    parser.add_argument("--input_file", type=str, default="final_2000.json")
    parser.add_argument("--img_dir", type=str)
    # output
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_file", type=str, default="llama3.json")
    # model
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="llm engine",
        choices=["gpt-3.5-turbo", "claude-2", "gpt4", "gpt-4-0613", "bard", 'qwen2_vl', 'QVQ','gpt-4o-mini','gpt-4o','deepseek_v3','qwen2_5','llama3'],
    )
    parser.add_argument(
        "--rerun", action="store_true", help="rerun answer extraction for all problems"
    )
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_file.replace('.json','') + "_" + args.mode + ".json"
    output_file = os.path.join(args.output_dir, output_file)

    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    # load model
    if args.model == "qwen2_vl":
        model = qwen2.Qwen2VLInference()
    elif args.model == "gpt-4-turbo" or args.model == "gpt-4o-mini" or args.model == "gpt-4o":
        model = gpt.GPT_Model(args.model, '1', patience=5)
    elif args.model == "llava-cot":
        model = llava_cot.LLavaCOTInference()
    elif args.model == "InternVL2_5-8B" or args.model == "InternVL2_5-78B" or args.model == "InternVL2_5-38B" or args.model == "InternVL2_5-26B":
        model = internVL2_5_78.InternVL2Inference(args.model)
    elif args.model == "QVQ":
        model = QVQ.QVQModel()
    elif args.model == "deepseek_v3":
        model = Deepseek.DeepSeek_v3()
    elif args.model == "qwen2_5":
        model = Qwen2_5.Qwen2_5()
    elif args.model == "llama3":
        model = llama3.Llama3()
    print(f"Model loaded.")
    # build final test pid list
    test_pids = data.keys()
    print("\nNumber of test problems in total:", len(test_pids))
    temp_results = results.copy()
    # tqdm, enumerate results
    for _, problem in tqdm(data.items()):
        pid = problem["id"]

        results[pid] = problem 
        if args.mode =='Text_Centric' or args.mode =='Text_Limited' or args.mode =='Vision_Dense':
            image = args.img_dir + '/' + problem['img_1']
        elif args.mode =='Vision_Centric':
            image = args.img_dir + '/' + problem['img_5']
        elif args.mode =='Vision_Primary':
            image = args.img_dir + '/' + problem['img_6']
        else:
            image = None

        problem["solution"].insert(0, '') # 还是得加上没有解决方案的情况
        for it, step_text in enumerate(problem["solution"]):
            if pid in temp_results:
                if temp_results[pid].get(f"response_step{it}")  and temp_results[pid][f"response_step{it}"] != "None":
                    print(f"Results already exist for {pid} _ {it}.")
                    results[pid]["query_"+ str(it)] = temp_results[pid]["query_"+ str(it)]
                    results[pid][f"response_step{it}"] = temp_results[pid][f"response_step{it}"]
                    continue

            query = create_one_query_cot(
                problem=problem,
                step_text = step_text,
                mode=args.mode
            )
            print(f"\nGenerating response for {pid} _ {it}...")
            try:
                if problem["answer"] == "None":
                    response = problem["analyse"]
                else:
                    response = model.get_response(image, query)
                # print(f"Response: {response}")
                results[pid]["query_"+ str(it)] = query
                response_label = "response_step" + str(it)
                results[pid][response_label] = response
                results[pid]['solution_step'] = len(problem["solution"])
            except Exception as e:
                print(e)
                print(f"Error in extracting answer for {pid}")
                results[pid]["error"] = e
        # save results
        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")

