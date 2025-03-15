import argparse
import os
from openai import OpenAI
import json
import re

from tqdm import tqdm
# client = OpenAI()



output_filename = "./batch_responses.jsonl"
# batch_input_file = client.files.create(
#   file=open("./models/batchoutput.jsonl", "rb"),
#   purpose="batch"
# )

def save_json(json_text, output_filename):
    """
    Extract 'content' values using regular expressions and save to a JSONL file.

    Parameters:
    - json_text (str): A string containing multiple JSON objects separated by newlines.
    - output_filename (str): The filename for the output JSONL file.
    """
    pattern = r'"content\\": \s*\\"([^\"]*)\\'
    contents = re.findall(pattern, json_text)
    
    with open(output_filename, 'w') as file:
        for content in contents:
            # Create a dictionary with the content
            content_dict = {"content": content}
            # Convert the dictionary to a JSON string and write it to the file
            json.dump(content_dict, file, ensure_ascii=False, indent=4)
            file.write('\n')  # Write a newline to separate entries




# bb = client.batches.retrieve(batch_job.id)
# # 检索批处理作业
# batch = client.batches.retrieve(batch_job.id)

import time

def check_batch_status(batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Current batch status: {status}")

        if status in ["validating", "in_progress", "finalizing"]:
            print("Batch is still processing. Waiting...")
            time.sleep(10)  # wait for 1 minute before checking again
        elif status == "completed":
            print("Batch completed successfully.")
            file_response = client.files.content(batch.output_file_id)
            return file_response.text
            save_json(file_response.text, output_filename)
            # print(file_response.text)  # Print the result
            break
        elif status == "failed":
            print("Batch failed during processing.")
            break
        elif status == "expired":
            print("Batch expired.")
            break
        elif status == "cancelled":
            print("Batch has been cancelled.")
            break
        elif status == "cancelling":
            print("Batch is currently being cancelled.")
        else:
            print("Unexpected status.")
            break

# Replace 'your_batch_id' with your actual batch ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument(
        "--data_dir", type=str, default="./data"
    )  # /home/dataset/AIMath/MathVista/data
    parser.add_argument("--mode", type=str, default="plus") # ["v1", "v2", "v3", "v4", "v5", "v6"]
    parser.add_argument("--input_dir", type=str, default="batch_data/output_gpt4_plus") #  output_gpt4_plus
    # output
    parser.add_argument("--output_dir", type=str, default="./my_results/10demo")
    parser.add_argument("--output_file", type=str, default="output_gpt4_plus.json")

    args = parser.parse_args()

    client = OpenAI(
                api_key = 'sk-proj-gLMxW0',
                base_url= 'https://api.jzx.ai/openai/v1')
    ori_data = json.load(open(args.data_dir + "/CZ_text_plus.json", "r"))

    os.makedirs(args.output_dir, exist_ok=True)
    for file in tqdm(os.listdir(args.input_dir), unit="file", desc="Processing files"):
        batch_input_file = client.files.create(file=open(args.input_dir+'/'+file, "rb"),purpose="batch")
        input_id= file.split(".")[0]
        tqdm.write(f"Currently processing: {input_id}")

        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "nightly eval job"
            }
        )
        response = check_batch_status(batch_job.id)
        pattern = r'"content\\": \s*\\"([^\"]*)\\'
        contents = re.findall(pattern, response)
        try:
            res = bytes(contents[0], 'utf-8').decode('unicode_escape')
        except:
            pass
        try:
            res = bytes(res, 'utf-8').decode('unicode_escape')
        except:
            pass
        ori_data[input_id]["response"] = res
        json.dump(ori_data, open(args.output_dir + "/" + args.output_file, "w"), ensure_ascii=False, indent=4)
