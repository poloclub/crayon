import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertModel

class BertMNLI(nn.Module):
    def __init__(self):
        super(BertMNLI, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.fc = nn.Linear(768, 3, bias=True)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids)
        output_2 = self.dropout(output_1.pooler_output)
        output = self.fc(output_2)
        return output