# shoplifting_detection/models/text_analyzer.py

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class TextAnalyzer(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def analyze(self, text):
        return self.forward(text)