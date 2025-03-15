from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import os



class Qwen2VLInference:
    def __init__(self, model_dir, min_pixels=256*28*28, max_pixels=1280*28*28):
        # Load the model in half-precision on the available device(s)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def get_response(self, image_path, user_prompt):
        # Load image
        if image_path == None:
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt} ]}]
        else:
            image = Image.open(image_path)
            # Prepare message with image and text prompt
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }]
        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0]
