import argparse, sys, time

import torch
import torch.nn as tnn

# Hyperparameters Begin
BATCH_SIZE = 64            # Number of independent sequences to process at once
BLOCK_SIZE = 256           # Maximum context length to consider for prediction
ITERATIONS = 5000          # Total training iterations
ITER_EVAL  = 500           # How often to report results
LEARNING_RATE = 3e-4       # Hyperparameter for training rate
NUM_EMBEDDINGS = 384       # Number of embedding layers
N_HEAD = 6                 # Number of self-attention heads
N_LAYER = 6                # Number of layers deep in model
DROPOUT = 0.2              # How often to dropout layers
# Hyperparameters End

VALIDATION_PERCENTAGE = 10 # Amount of data to hold back for validation
DEVICE = 'cpu'             # Which device to train with
if torch.cuda.is_available():
    DEVICE = 'cuda'

#
# Papers
#   Attention is all you need (Transformers)
#        https://arxiv.org/abs/1706.03762
#   Deep Residual Learning for Image Recognition
#        https://arxiv.org/abs/1512.03385
#   Dropout: A Simple Way to Prevent Neural Networks from Overfitting
#        https://jmlr.org/papers/v15/srivastava14a.html
#

def main():
    args = parse_args()
    print('Training on device "{}"'.format(DEVICE))

    # Read and tokenize input data
    text = read_input(args.input_file)
    (vocab_size, encode, decode) = generate_encoder_decoder(text)
    data = torch.tensor(encode(text), dtype=torch.long)

    # Divide data into training and validation
    n = int((1 - VALIDATION_PERCENTAGE / 100) * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Create model
    model = BigramLanguageModel(vocab_size, N_LAYER, N_HEAD)
    model.to(DEVICE)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    start = time.perf_counter()
    for step in range(ITERATIONS):
        # Evaluate progress
        if step % ITER_EVAL == 0:
            stop = time.perf_counter()
            losses = estimate_loss(model, train_data, val_data, 100)
            print('Training, step: {}, t loss: {}, v loss: {}, time: {} s'.format(
                step, losses[0], losses[1], stop - start))
            output = decode(generate_tokens(model, 100))
            print('Sample output: {}'.format(output))
            start = time.perf_counter()

        # Get a random sample from training data
        (bx, by) = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)

        # Evaluate loss and optimize
        logits, loss = model(bx, by)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def parse_args():
    parser = argparse.ArgumentParser(
        prog='gpt',
        description='Simple nanoGPT implementation, based on https://www.youtube.com/watch?v=kCc8FmEb1nY')
    parser.add_argument('input_file')
    return parser.parse_args()

def read_input(fname):
    with open(fname, 'r') as f:
        text = f.read()
    print('Read input file {} with {} characters'.format(fname, len(text)))
    return text

def generate_encoder_decoder(text):
    """Simple tokenizer. Character-based, not sub-word based.
       GPT-2 uses tiktoken (https://github.com/openai/tiktoken). 
       Google uses SentencePiece (https://github.com/google/sentencepiece).
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print('Vocab size {}, chars: {}'.format(vocab_size, ''.join(chars)))
    stoi = { ch:i for i, ch in enumerate(chars) }
    itos = { i:ch for i, ch in enumerate(chars) }
    # Define encoder/decoder, this one just translates each char to an int
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return (vocab_size, encode, decode)

def get_batch(data, block_size, batch_size):
    """Returns a batch of data from the data set provided"""
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in idx])
    y = torch.stack([data[i+1 : i+block_size+1] for i in idx])
    return (x.to(DEVICE), y.to(DEVICE))

def generate_tokens(model, size):
    idx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
    return model.generate(idx, size)[0].tolist()

@torch.no_grad()
def estimate_loss(model, train, val, iterations):
    out = {}
    model.eval()
    for i, data in enumerate((train, val)):
        losses = torch.zeros(iterations)
        for k in range(iterations):
            (x, y) = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            (logits, loss) = model(x, y)
            losses[k] = loss.item()
        out[i] = losses.mean()
    model.train()
    return out

class Head(tnn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        n_embd = NUM_EMBEDDINGS
        # Each token directly reads off the logits for the next token from a lookup table
        self.key   = tnn.Linear(n_embd, head_size, bias=False)
        self.query = tnn.Linear(n_embd, head_size, bias=False)
        self.value = tnn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = tnn.Dropout(DROPOUT)
    
    def forward(self, x):
        (B,T,C) = x.shape
        k = self.key(x)                                               # (B,T,C)
        q = self.query(x)                                             # (B,T,C)
        # Compute attention scores, affinities between tokens
        wei = q @ k.transpose(-2, -1) * C**-0.5                       # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = tnn.functional.softmax(wei, dim=-1)                     # (B,T,T)
        wei = self.dropout(wei)
        # Perform weighted aggregation of the values
        v = self.value(x)                                             # (B,T,C)
        out = wei @ v                                                 # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(tnn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = tnn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = tnn.Linear(NUM_EMBEDDINGS, NUM_EMBEDDINGS)
        self.dropout = tnn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(tnn.Module):
    """A linear layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = tnn.Sequential(
            tnn.Linear(n_embd, 4 * n_embd), 
            tnn.ReLU(),
            tnn.Linear(4 * n_embd, n_embd), 
            tnn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(tnn.Module):
    """Transformer block: communication, followed by computation"""
    def __init__(self, n_embd, n_head): # Embedding dimension, number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = tnn.LayerNorm(n_embd)
        self.ln2 = tnn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(tnn.Module):
    def __init__(self, vocab_size, n_layer, n_head):
        super().__init__()
        n_embd = NUM_EMBEDDINGS
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = tnn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = tnn.Embedding(BLOCK_SIZE, n_embd)
        self.blocks = tnn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = tnn.LayerNorm(n_embd)
        self.lm_head  = tnn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        (B, T) = idx.shape
        # Index and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx)          # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb                              # (B,T,C)
        x = self.blocks(x)                                 # (B,T,C)
        x = self.ln_final(x)                               # (B,T,C)
        logits = self.lm_head(x)                           # (B,T,V_size)

        if targets is None:
            loss = None
        else:
            # Reshape and compute loss
            (B, T, C) = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = tnn.functional.cross_entropy(logits, targets)

        return (logits, loss)

    def generate(self, idx, max_new_tokens):
        # Index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop index to last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get predictions
            (logits, loss) = self(idx_cond)
            # Only for the last time step, transform to (B,C)
            logits = logits[:, -1, :] 
            # Apply softmax to get probabilities
            probs = tnn.functional.softmax(logits, dim=-1)
            # Sample from distribution, produces (B,1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence, produces (B,T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    main()
