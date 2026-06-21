import torch
import torch.nn as nn
from transformers import BertModel

class MultiTaskEmailModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)

        self.spam_head = nn.Linear(768, 3)
        self.priority_head = nn.Linear(768, 3)
        self.intent_head = nn.Linear(768, 6)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_vector = bert_outputs.pooler_output

        cls_vector = self.dropout(cls_vector)

        spam_logits = self.spam_head(cls_vector)
        priority_logits = self.priority_head(cls_vector)
        intent_logits = self.intent_head(cls_vector)

        return {
            "spam": spam_logits,
            "priority": priority_logits,
            "intent": intent_logits
        }