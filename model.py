import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.cuda.amp import GradScaler, autocast

class JadeModel(nn.Module):
    def __init__(self, load_model_path=None):
        super(JadeModel, self).__init__()
        # GPT-like Transformer architecture
        self.vocab_size = 512  # Character-level tokenization (ASCII range)
        self.embedding_dim = 768  # GPT-like embedding dimension
        self.num_heads = 12  # Number of attention heads
        self.num_layers = 12  # Number of transformer layers
        self.max_position_embeddings = 512  # Maximum sequence length

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_position_embeddings, self.embedding_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Load model state if path is provided
        if load_model_path and os.path.exists(load_model_path):
            self.load_model(load_model_path)
            print(f"Model loaded from {load_model_path}")

    def forward(self, input_ids):
        # Truncate input_ids if longer than max_position_embeddings
        if input_ids.size(1) > self.max_position_embeddings:
            input_ids = input_ids[:, -self.max_position_embeddings:]
        
        # Create position ids for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Embedding lookup
        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output layer
        x = self.fc(x)
        return x

    def generate_response(self, input_text, initial_temperature=0.85, top_p=0.8, repetition_penalty=1.4, max_token_frequency=2, max_length=50, min_response_length=5):
        # Convert input_text to token ids
        input_ids = self.tokenize(input_text)
        if len(input_ids) > self.max_position_embeddings:
            input_ids = input_ids[-self.max_position_embeddings:]  # Truncate if too long
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        generated_tokens = input_ids.copy()  # Start with input tokens to use as context
        temperature = initial_temperature
        recent_tokens = list(input_ids[-10:])  # Expanded recent tokens window to 10

        with torch.no_grad(), autocast():
            for _ in range(max_length):  # Generate up to max_length more tokens
                output = self.forward(input_tensor)
                logits = output[:, -1, :]  # Consider only the last token's logits
                logits = logits / (temperature + 1e-2)  # Apply temperature for sampling diversity

                # Apply repetition penalty
                for token in set(generated_tokens):
                    if generated_tokens.count(token) > 1:
                        logits[0, token] /= (repetition_penalty + generated_tokens.count(token) * 0.02)  # Frequency-based scaling for penalty

                # Dynamic Nucleus (top-p) sampling with adjusted threshold
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
                top_p_mask = cumulative_probs < top_p
                top_p_logits = sorted_logits[top_p_mask]
                top_p_indices = sorted_indices[top_p_mask]

                if len(top_p_logits) > 1:
                    top_p_probs = self.softmax(top_p_logits)
                    sampled_token = top_p_indices[torch.multinomial(top_p_probs, num_samples=1).item()].item()
                else:
                    sampled_token = sorted_indices[0, 0].item()  # Fallback to the most probable token if none pass the top-p threshold
                
                # Add token and update state
                generated_tokens.append(sampled_token)
                if len(recent_tokens) > 10:
                    recent_tokens.pop(0)  # Maintain a window of recent tokens to suppress
                
                # Update input tensor to include the generated token
                input_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(self.device)
                
                # Gradually decrease temperature to reduce randomness more smoothly
                temperature = max(0.75, temperature * 0.98)

        response = self.detokenize(generated_tokens[len(input_ids):])  # Exclude the input from the response
        return response if len(response.strip()) > 0 else None

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    # Placeholder tokenization method (to be replaced with optimized tokenizer)
    def tokenize(self, text):
        return [ord(c) for c in text]

    # Placeholder detokenization method (to be replaced with optimized tokenizer)
    def detokenize(self, tokens):
        return ''.join([chr(t) for t in tokens])
