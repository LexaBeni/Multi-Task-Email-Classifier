import torch 
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class MultiTaskEmailDataset(Dataset):
    def __init__(self, pandas_df, model_name="bert-base-uncased", max_len=64):
        super().__init__()
        self.df = pandas_df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        text = str(line['text'])

        tokens = self.tokenizer(text, padding = "max_length", truncation = True, max_length = 64, return_tensors="pt")

        item = {key: tensor.squeeze(0) for key, tensor in tokens.items()}

        item["label_spam"] = torch.tensor(line["label"], dtype=torch.long)
        item["label_priority"] = torch.tensor(line["priority"], dtype=torch.long)
        item["label_intent"] = torch.tensor(line["intent"], dtype=torch.long)

        return item