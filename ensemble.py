import torch.nn as nn
import torch


class Ensemble_Model(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble_Model, self).__init__()
        self.Text_Model = modelA
        self.Image_Model = modelB
        self.hidden_layer1 = nn.Linear(544512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 4)



    def forward(self, text_input_ids, text_attention_mask, image_embedding):
        text_input = self.Text_Model(text_input_ids, text_attention_mask)
        image_input = self.Image_Model(image_embedding)
        concat_input = torch.cat([torch.flatten(text_input), torch.flatten(image_input)])
        output = self.output_layer(self.hidden_layer2(self.hidden_layer1(concat_input)))
        return output

