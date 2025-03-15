import base64
import os
import cv2
import json
import time
import pickle
import openai
import re
from word2number import w2n
from openai import OpenAI


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def read_csv(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            data.append(line.strip())
    return data


def read_pandas_csv(csv_path):
    # read a pandas csv sheet
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(file):
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data, path):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_array_img(path, image):
    cv2.imwrite(path, image)


def contains_digit(text):
    # check if text contains a digit
    if any(char.isdigit() for char in text):
        return True
    return False


def contains_number_word(text):
    # check if text contains a number word
    ignore_words = ["a", "an", "point"]
    words = re.findall(
        r"\b\w+\b", text
    )  # This regex pattern matches any word in the text
    for word in words:
        if word in ignore_words:
            continue
        try:
            w2n.word_to_num(word)
            return True  # If the word can be converted to a number, return True
        except ValueError:
            continue  # If the word can't be converted to a number, continue with the next word

    # check if text contains a digit
    if any(char.isdigit() for char in text):
        return True

    return False  # If none of the words could be converted to a number, return False


def contains_quantity_word(text, special_keep_words=[]):
    # check if text contains a quantity word
    quantity_words = [
        "most",
        "least",
        "fewest" "more",
        "less",
        "fewer",
        "largest",
        "smallest",
        "greatest",
        "larger",
        "smaller",
        "greater",
        "highest",
        "lowest",
        "higher",
        "lower",
        "increase",
        "decrease",
        "minimum",
        "maximum",
        "max",
        "min",
        "mean",
        "average",
        "median",
        "total",
        "sum",
        "add",
        "subtract",
        "difference",
        "quotient",
        "gap",
        "half",
        "double",
        "twice",
        "triple",
        "square",
        "cube",
        "root",
        "approximate",
        "approximation",
        "triangle",
        "rectangle",
        "circle",
        "square",
        "cube",
        "sphere",
        "cylinder",
        "cone",
        "pyramid",
        "multiply",
        "divide",
        "percentage",
        "percent",
        "ratio",
        "proportion",
        "fraction",
        "rate",
    ]

    quantity_words += special_keep_words  # dataset specific words

    words = re.findall(
        r"\b\w+\b", text
    )  # This regex pattern matches any word in the text
    if any(word in quantity_words for word in words):
        return True

    return False  # If none of the words could be converted to a number, return False


def is_bool_word(text):
    if text in [
        "Yes",
        "No",
        "True",
        "False",
        "yes",
        "no",
        "true",
        "false",
        "YES",
        "NO",
        "TRUE",
        "FALSE",
    ]:
        return True
    return False


def is_digit_string(text):
    # remove ".0000"
    text = text.strip()
    text = re.sub(r"\.0+$", "", text)
    try:
        int(text)
        return True
    except ValueError:
        return False


def is_float_string(text):
    # text is a float string if it contains a "." and can be converted to a float
    if "." in text:
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


def copy_image(image_path, output_image_path):
    from shutil import copyfile

    copyfile(image_path, output_image_path)


def copy_dir(src_dir, dst_dir):
    from shutil import copytree

    # copy the source directory to the target directory
    copytree(src_dir, dst_dir)


import PIL.Image as Image


def get_image_size(img_path):
    img = Image.open(img_path)
    width, height = img.size
    return width, height


def get_chat_response(
    promot,
    api_key,
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=256,
    n=1,
    patience=10000000,
    sleep_time=0,
):
    messages = [
        {"role": "user", "content": promot},
    ]
    # print("I am here")
    while patience > 0:
        patience -= 1
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            if n == 1:
                prediction = response["choices"][0]["message"]["content"].strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [
                    choice["message"]["content"].strip()
                    for choice in response["choices"]
                ]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(promot) * 0.9)
                new_start = len(promot) - new_size
                promot = promot[new_start:]
                messages = [
                    {"role": "user", "content": promot},
                ]

            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# build gpt class
class GPT_Model:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        api_key="",
        temperature=0,
        max_tokens=1024,
        n=1,
        patience=1000000,
        sleep_time=0,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.patience = patience
        self.sleep_time = sleep_time
        # Init OpenAI client instance
        self.openai_api_key = api_key  # os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise Exception("Missing OpenAI API key.")
        self.headers_override = {
            "Content-Type": "application/json",
            "Authorization": "332fcee8ed504e34a1a06a7a770621ad",  # os.getenv("AZURE_API_KEY")
        }
        self.private_base_url = "https://jzx-ai-api-gateway.azure-api.net/v1"  # os.getenv("OPENAI_BASE_URL")
        if os.getenv("OVERRIDE_HEADER", "0") == "1":
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.private_base_url,
                default_headers=self.headers_override,
            )
        else:
            raise Exception(
                "Please set OVERRIDE_HEADER=1 and your AZURE_API_KEY in .env file"
            )

    def get_response(self, image_path, user_prompt):
        patience = self.patience
        max_tokens = self.max_tokens
        ENCODING = "utf-8"
        if image_path !='':
            image_path = image_path.replace("./data", "/train/dataset/AI4Math/MathVista")
            base64_image = encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",  #  "https://kexue.fm/usr/uploads/2022/07/3685027055.jpeg"
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
        while patience > 0:
            patience -= 1
            try:
                # print("self.model", self.model)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    n=self.n,
                )
                if self.n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
                else:
                    prediction = [
                        choice.message.content.strip() for choice in response.choices
                    ]
                    if prediction[0] != "" and prediction[0] != None:
                        return prediction

            except Exception as e:
                if "limit" not in str(e):
                    print(e)
                if "Please reduce the length of the messages or completion" in str(e):
                    max_tokens = int(max_tokens * 0.9)
                    print("!!Reduce max_tokens to", max_tokens)
                if max_tokens < 8:
                    return ""
                if "Please reduce the length of the messages." in str(e):
                    print("!!Reduce user_prompt to", user_prompt[:-1])
                    return ""
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""
