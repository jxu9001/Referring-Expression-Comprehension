import torch.nn as nn
from tf import *
from vf import *
from ensemble import *



modelA = text_tf("distilbert-base-uncased")
modelB = image_tf("google/vit-base-patch16-224-in21k")
Ensemble = Ensemble_Model(modelA, modelB)


example_A = modelA.tokenize("A man and a pen.")

dataset = load_dataset("huggingface/cats-image")
example_B = modelB.patcher(dataset["test"]["image"][0])
Ensemble.forward(example_A, example_B)




#example_A = modelA.tokenize("A man and a pen.")
#input_ids_A = torch.tensor(example_A['input_ids'])
#attention_mask_A = torch.tensor(example_A['attention_mask'])
#modelA.forward(input_ids_A, attention_mask_A)


#example_B = modelB.tokenize("My pen is blue and my friend is you.")
#input_ids_B = torch.tensor(example_B['input_ids']).unsqueeze(0)
#attention_mask_B = torch.tensor(example_B['attention_mask']).unsqueeze(0)
#modelB.forward(input_ids_B, attention_mask_B)

