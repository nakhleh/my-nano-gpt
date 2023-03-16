import argparse, sys, time

import torch
import torch.nn as tnn

VALIDATION_PERCENTAGE = 10 # Amount of data to hold back for validation
BLOCK_SIZE = 8             # Maximum context length to consider for prediction
BATCH_SIZE = 4             # Number of independent sequences to process at once
LEARNING_RATE = 1e-3       # Hyperparameter for training rate
ITERATIONS = 100000        # Total training iterations
ITER_EVAL  = 10000         # How often to report results
DEVICE = 'cpu'             # Which device to train with
if torch.cuda.is_available():
    DEVICE = 'cuda'

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
    model = BigramLanguageModel(vocab_size)
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

class BigramLanguageModel(tnn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = tnn.Embedding(vocab_size, vocab_size, device=DEVICE)
    
    def forward(self, idx, targets=None):
        # Index and targets are both (B,T) tensors of integers, output is (B,T,C)
        logits = self.token_embedding_table(idx)

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
            # Get predictions
            (logits, loss) = self(idx)
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
