from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import base64
import io
from PIL import Image

def encode_image(image_path):
    # 打开并转换图片为PNG格式
    with Image.open(image_path) as img:
        # 转换为RGB模式（如果是RGBA，保持不变）
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        # 创建内存缓冲区存储PNG图片
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        # 获取base64编码
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
class Claude_Model:
    def __init__(self):
        self.anthropic = Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
        )

    def get_response(self, image_path, user_prompt):
        if image_path:
            base64_image = encode_image(image_path)
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]

        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages = messages
        )

        return response.content[0].text


if __name__ == "__main__":
    claude = Claude_Model()
    IMAGE_PATH = "/home/ecs-user/nas_original_data/csh/DATA/img_739/68b6b198-0730-408e-a6c6-387d0ced4dc1/68b6b198-0730-408e-a6c6-387d0ced4dc1_1.png"
    PROMPT = "describe this image"
    print(claude.get_response(IMAGE_PATH, PROMPT))

