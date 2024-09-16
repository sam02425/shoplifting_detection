# shoplifting_detection/models/text_generator.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator(torch.nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, video_features):
        # Convert video features to text prompt
        prompt = f"Describe a scene where a person is "
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(video_features.device)

        # Generate text based on video features
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1,
                                     no_repeat_ngram_size=2, top_k=50, top_p=0.95)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate(self, video_features):
        return self.forward(video_features)