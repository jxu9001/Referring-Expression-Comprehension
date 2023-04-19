import torch.nn as nn
import torch


class Ensemble_Model(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble_Model, self).__init__()
        self.Text_Model = modelA
        self.Image_Model = modelB
        self.linear_bbox = nn.Linear(768, 4)
    
    def forward(self, text_embedding, image_embedding):
        text_output = self.Text_Model(torch.tensor(text_embedding['input_ids']).unsqueeze(0), torch.tensor(text_embedding['attention_mask']).unsqueeze(0))
        image_output = self.Image_Model(image_embedding)
        
        # Apply learned weights to the subnetwork outputs
        text_output = text_output.unsqueeze(2).expand(-1, -1, 768)
        print(text_output.shape, image_output.shape)
        # Concatenate the subnetwork outputs along the second dimension
        ensemble_output = torch.cat((text_output, image_output), dim=1)

        pred_boxes = self.linear_bbox(ensemble_output).sigmoid()
        
        return pred_boxes
