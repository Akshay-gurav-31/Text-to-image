import os
from flask import Flask, render_template, request
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from io import BytesIO
import base64

load_dotenv()
app = Flask(__name__)

client = InferenceClient(token=os.getenv("HF_TOKEN"))

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    prompt = ""
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev",
                guidance_scale=7.5,
                negative_prompt="low quality, blurry, distorted, bad anatomy"
            )
            buf = BytesIO()
            image.save(buf, format="PNG")
            image_data = base64.b64encode(buf.getvalue()).decode()
    return render_template("index.html", image_data=image_data, prompt=prompt)

if __name__ == "__main__":
    app.run(debug=True)
