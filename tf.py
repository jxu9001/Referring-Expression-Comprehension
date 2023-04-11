from datasets import load_dataset, Dataset, DatasetDict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig



class text_tf(nn.Module):
    def __init__(self, model_choice):
        super(text_tf, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_choice)
        self.model = AutoModel.from_pretrained(model_choice, config=AutoConfig.from_pretrained(model_choice,
                                                                                               output_attention=True,
                                                                                               output_hidden_state=True)
                                               )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0][:,0,:].view(-1,768)
        return last_hidden_state


    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, max_length=512,padding="max_length")




