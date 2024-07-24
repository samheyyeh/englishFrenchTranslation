#!/usr/bin/env python3

from flask import Flask
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

def translate(text):
    tokenizer = AutoTokenizer.from_pretrained("rajbhirud/eng-to-fra-model")
    model = AutoModelForSeq2SeqLM.from_pretrained("rajbhirud/eng-to-fra-model")

    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)

    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=64, num_beams=2)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

if __name__ == "__main__":
    print("English to French Translation")
    while True:
        text = input("Enter English text (control + c to exit): ")
        translation = translate(text)
        print("French translation:", translation)
