from app.services.preprocessing import clean_text
import torch

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def predict(text, model, tokenizer) :
    text = clean_text(text)
    tokenized_text = tokenizer(text, max_length = 128, truncation = True, return_tensors = 'pt')
    with torch.no_grad() :
        output = model(tokenized_text['input_ids'], tokenized_text['attention_mask']) #evaluates the model with this text in model_loader.py
        probs = output[0] #removes the batch dimension converts to a tensor with 6 values
        idx = torch.argmax(probs).item() # .item() converts tensor to plain python int
        confidence = probs[idx].item()

    return {
        "label" : labels[idx],
        "confidence" : confidence,
        "uncertain" : confidence < 0.5
    }