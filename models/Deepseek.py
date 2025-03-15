# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import base64

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
class DeepSeek_v3:
    def __init__(self):
        self.client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    def get_response(self, image_path,user_prompt):
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
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content


