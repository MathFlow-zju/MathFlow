import base64

def encode_image(image_path, output_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    with open(output_path, "w") as output_file:
        output_file.write(encoded_string)

# Example usage
encode_image("../JZX_Verse/img_788/0b14a59f-99fb-4323-a940-47e13d88b63b/0b14a59f-99fb-4323-a940-47e13d88b63b_1.png", "./encoded_image.txt")
