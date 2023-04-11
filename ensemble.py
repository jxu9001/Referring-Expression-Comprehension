import torch.nn as nn
import torch


class Ensemble_Model(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble_Model, self).__init__()
        self.Text_Model = modelA
        self.Image_Model = modelB

    def forward(self, text_embedding, image_embedding):
        text_output = self.Text_Model(torch.tensor(text_embedding['input_ids']).unsqueeze(0), torch.tensor(text_embedding['attention_mask']).unsqueeze(0))
        image_output = self.Image_Model(image_embedding)
        #ensemble_input = torch.concat([text_output, image_output], 1)
        print(text_output.shape, image_output.shape)
        return