import torch.nn as nn
from tf import *
from vf import *
from ensemble import *
import torch
from torchvision import transforms
from collections import defaultdict
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F



def resize224(image_width, image_height, x, y , w, h):

    y_ = image_height
    x_ = image_width
    targetSize = 224
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    (origLeft, origTop, origRight, origBottom) = (x, y, x + w, y + h)
    x = int(origLeft * x_scale)
    y = int(origTop * y_scale)
    xmax = int(origRight * x_scale)
    ymax = int(origBottom * y_scale)

    return [x, y, xmax-x, ymax-y]


dataset = load_dataset("imagefolder", data_dir="./sample_dataset")

mps_device = torch.device("mps")
modelA = text_tf("distilbert-base-uncased").to(mps_device)
modelB = image_tf("google/vit-base-patch16-224-in21k").to(mps_device)
Ensemble = Ensemble_Model(modelA, modelB).to(mps_device)
parameters = filter(lambda p: p.requires_grad, Ensemble.parameters())
optimizer = torch.optim.Adam(parameters , lr =0.006)



train_inputs = []
idx = 0
for sample in dataset['train']:
    processed_text = modelA.tokenize(sample['expression'])
    processed_image = modelB.patcher(sample['image'])
    resized_coordinate = resize224(sample['img_width'], sample['img_height'], sample['x'], sample['y'], sample['w'], sample['h'])
    train_inputs.append(
        {
            'input_ids': torch.tensor(processed_text['input_ids'], device="mps"),
            'attention_mask': torch.tensor(processed_text['attention_mask'], device="mps"),
            'image_embedding': processed_image.to(mps_device),
            'BBox': torch.tensor(resized_coordinate, device="mps")
        }
    )

train_loader = DataLoader(train_inputs, batch_size=1, shuffle=True)


loss = 0
for _ in range(5):
    for batch_index, batch_inputs in enumerate(train_loader):
        input_ids = batch_inputs['input_ids']
        input_masks = batch_inputs['attention_mask']
        input_images = batch_inputs['image_embedding']
        input_images['pixel_values'] = input_images['pixel_values'][0]
        batch_output = Ensemble(input_ids, input_masks, input_images)
        print(batch_output, batch_inputs['BBox'])
        loss_BBox = F.l1_loss(torch.reshape(batch_output,(1,4)), batch_inputs['BBox'], reduction="mean")
        loss_BBox = loss_BBox.sum()
        torch.nn.utils.clip_grad_norm_(parameters=Ensemble.parameters(), max_norm=1)
        optimizer.zero_grad()
        loss_BBox.backward()
        optimizer.step()



'''


example_A = modelA.tokenize("There are two cats.")
dataset = load_dataset("huggingface/cats-image")
example_B = modelB.patcher(dataset["test"]["image"][0])
print(example_B['pixel_values'].shape)
Ensemble(text_input_ids = torch.tensor(example_A['input_ids']), text_attention_mask  = torch.tensor(example_A['attention_mask']), image_embedding = example_B)
'''