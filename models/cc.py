import anthropic
import base64

client = anthropic.Anthropic(
    api_key="",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

IMAGE_PATH = "/home/ecs-user/nas_original_data/csh/DATA/img_739/68b6b198-0730-408e-a6c6-387d0ced4dc1/68b6b198-0730-408e-a6c6-387d0ced4dc1_1.png"
PROMPT = "describe this image"

base64_image = encode_image(IMAGE_PATH)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
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
                "text": PROMPT
            }
        ]
    }]
)

print(message.content[0].text)