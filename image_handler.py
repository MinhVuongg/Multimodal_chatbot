from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64

def convert_bytes_to_base64(image_bytes):
    encode_string = base64.b64encode(image_bytes).decode("utf-8")  # Properly encode the bytes to base64
    return "data:image/jpeg;base64," + encode_string  # Correct data URL format


def handle_image(image_bytes, user_message):
    
    chat_handler = Llava15ChatHandler(clip_model_path="./models/llava/mmproj-model-f16.gguf")
    llm = Llama(
        model_path ="./models/llava/ggml-model-q5_k.gguf",
        chat_handler = chat_handler ,
        logits_all=True,
        n_ctx=1024
    )
    image_base64 = convert_bytes_to_base64(image_bytes)
    
    output= llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "Bạn là trợ lý mô tả hình ảnh một cách hoàn hảo"},
            {
                "role": "user",
                "content" : [
                    {"type": "image_url", "image_url":{"url": image_base64}},
                    {"type": "text", "text": user_message}
                ]
            }
        ]
    )
    print(output)
    # return output["choice"][0]["message"]["content"]    
    
    
        # Access the message content based on the correct structure
    try:
        # Assuming the correct key is 'choices' and the structure is known
        return output["choices"][0]["message"]["content"]
    except KeyError as e:
        # Print error and output for debugging
        print(f"KeyError: {e}")
        print("Output structure:", output)
        raise

# Use this function in your Streamlit app to see the output

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encode_string = base64.b64encode(image_file.read()).decode("utf-8")  # Properly encode the bytes to base64
        return "data:image/jpeg;base64," + encode_string  # Correct data URL format

if __name__ == "__main__":
    image_path = "./images/llama/jpg"
    image_base64 = convert_image_to_base64(image_path)
    with open("image.txt", "w") as f:
        f.write(image_base64)