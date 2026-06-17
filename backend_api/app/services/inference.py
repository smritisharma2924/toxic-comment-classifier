from app.services.preprocessing import clean_text
import torch

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THREAT_PATTERNS = [
    'make you disappear',
    'you should be careful',
    'watch your back',
    'you will regret',
    'something bad will happen',
    'you wont be around',
    'make you pay',
    'come for you',
    'hunt you down',
    'you are finished',
    'your days are numbered',
    'wont end well for you',
    'you better run',
    'you better hide',
    'youll be sorry',
    'going to make you suffer',
]

def check_threat_patterns(text):
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in THREAT_PATTERNS):
        return True
    return False

def predict(text, model, tokenizer) :
    text = clean_text(text)
    tokenized_text = tokenizer(text, max_length = 128, truncation = True, return_tensors = 'pt')
    with torch.no_grad() :
        output = model(tokenized_text['input_ids'], tokenized_text['attention_mask']) #evaluates the model with this text in model_loader.py
        probs = output[0] #removes the batch dimension converts to a tensor with 6 values
        idx = torch.argmax(probs).item() # .item() converts tensor to plain python int
        confidence = probs[idx].item()

    if check_threat_patterns(text):
        return {
            "label": "threat",
            "confidence": 0.85,
            "uncertain": False
        }

    return {
        "label" : labels[idx],
        "confidence" : confidence,
        "uncertain" : confidence < 0.5
    }