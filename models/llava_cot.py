import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os




class LLaVAModel:
    def __init__(self, model_id):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_response(self, image_path, user_prompt):
        if image_path == None:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            image = Image.open(image_path)
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        return self.processor.decode(outputs[0])

