import time
import google.generativeai as genai

# AIzaSyCbEuUGWmg7rsja9eJw_d3nXeaefEM6Vp4
# AIzaSyDLfNjo_Cn9CkwemLUpoG609JAEze_RPE4
genai.configure(api_key="AIzaSyDLfNjo_Cn9CkwemLUpoG609JAEze_RPE4", transport="rest") 

import httpx
import os
import base64
import cv2
import numpy as np
from PIL import Image



class GeminiModel:
    def __init__(self,patience=10000):
        self.model = genai.GenerativeModel(model_name = "gemini-1.5-pro-002")
        self.patience = patience

    def get_response(self, image_path, prompt):
        while self.patience > 0:
            self.patience -= 1
            try:
                if image_path is not None:
                    return self.model.generate_content([{'mime_type':'image/jpeg', 'data': self.encode_image(image_path)}, prompt]).text
                else:
                    return self.model.generate_content([prompt]).text
            except Exception as e:
                if "limit" not in str(e):
                    print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
