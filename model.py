from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#Loading the image captioning model and auto tokenizer
model = VisionEncoderDecoderModel.from_pretrained("jaimin/image_caption")
feature_extractor = ViTFeatureExtractor.from_pretrained("jaimin/image_caption")

tokenizer = AutoTokenizer.from_pretrained("jaimin/image_caption")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Predict function
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the form
        image = request.files['image']

        # Open the image
        img = Image.open(image.stream)

        # Convert the image to RGB mode if necessary
        if img.mode != "RGB":
            img = img.convert(mode="RGB")

        # Extract the pixel values and move them to the device
        pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate the caption using beam search
        output_ids = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Render the prediction on the home page
        return render_template('home.html', prediction=caption)

    except:
        # If there's an error, render the error message on the home page
        return render_template('home.html', error='Error processing image')
        
if __name__ == '__main__':
    app.run(debug=True)