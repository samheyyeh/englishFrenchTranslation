#!/usr/bin/env python3

from flask import Flask, render_template, request, url_for
from gtts import gTTS
import os
import torch
from transformers import MarianTokenizer, MarianMTModel
from datetime import datetime

app = Flask(__name__)

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)

    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=64, num_beams=2)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_route():
    text = request.form['text']
    translation = translate(text)

    tts = gTTS(translation, lang='fr')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_filename = f"translation_{timestamp}.mp3"
    save_path = os.path.join("static", audio_filename)
    tts.save(save_path)

    return render_template('index.html', translation=translation, audio_file=audio_filename)

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5555)

