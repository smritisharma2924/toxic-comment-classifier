import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
import os

class ToxicClassifier(nn.Module) :
    def __init__(self, bert) :
        super().__init__()
        # layers here
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768,6)

    def forward(self, input_ids, attention_mask) :
        # forward pass
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls = output[0][:,0,:]
        x = self.dropout(cls)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return probs
    
def load_model() :
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = ToxicClassifier(bert)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'distilbert_model.pt')
    weights = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(weights)
    model.eval()
    TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'distilbert_tokenizer')
    tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    return model,tokenizer