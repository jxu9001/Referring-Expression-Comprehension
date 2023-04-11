from transformers import AutoImageProcessor, ViTModel
import torch
import torch.nn as nn
from datasets import load_dataset


class image_tf(nn.Module):
    def __init__(self, model_choice):
        super(image_tf, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_choice)
        self.model = ViTModel.from_pretrained(model_choice)


    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        print(outputs.last_hidden_state)
        return outputs.last_hidden_state

    def patcher(self, image):
        return self.image_processor(image, return_tensors="pt")

