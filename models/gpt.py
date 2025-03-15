import time
from openai import OpenAI
import os
import base64


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
        max_tokens=512,
        n=1,
        patience=1000000,
        sleep_time=0,
        mode='azure',
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
        self.private_base_url = ""  # os.getenv("OPENAI_BASE_URL")
        self.mode = mode
        if os.getenv("OVERRIDE_HEADER", "0") == "1":
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.private_base_url,
                default_headers=self.headers_override,
            )
        else:
            if self.mode == 'azure':
                self.client = OpenAI(api_key = '')
            else:
                raise Exception(
                "Please set OVERRIDE_HEADER=1 and your AZURE_API_KEY in .env file"
            )

    def get_response(self, image_path, user_prompt):
        patience = self.patience
        max_tokens = self.max_tokens
        ENCODING = "utf-8"
        # image_path = image_path.replace("./data", "/train/dataset/AI4Math/MathVista")
        if image_path:
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
        # messages = [
        #     {"role": "user", "content": user_prompt},
        # ]
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
        return "None"
