import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 'cuda' for GPU, 'cpu' for CPU, makes things run a lot faster
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# This creates a small version of a tokenizer for us to use for the model
stoi = { ch:i for i,ch in enumerate(chars) }    # Line 1: builds a dictionary called stoi (string -> integer), mapping each character to a unique number
itos = { i:ch for i,ch in enumerate(chars) }    # Line 2: builds a dictionary called itos (integer -> string), maps integers back to characters (opposite of stoi)
encode = lambda s: [stoi[c] for c in s]         # Line 3: encode - turns a string into a list of numbers (tokens) using stoi dictionary
decode = lambda l: ''.join([itos[i] for i in l])# Line 4: decode - reverse function of line 3, turns a list of numbers back into the original string
print(encode("hii there"))                      # Line 5: prints the encoded list (list of numbers)
print(decode(encode("hii there")))              # Line 6: prints the decoded list (after ‘hii there’ is encoded into numbers, it decodes it back into letters in the original form)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # putting all the letters into a container of numbers
# If PyTorch is not installed, run:
# pip install torch

# Help understand to what extent the model is overfitting
# Help it to understand true Shakespeare text so that it is able to replicate it
n = int(0.9*len(data)) # keeping the first 90% as training size, the rest 10% is used for validation
train_data = data[:n] # Takes everything from the start of data up to index n (but not including n) - so 90%
val_data = data[n:] # Takes everything from n to the end (10%) for validation
# We know that len(train_data) + len(val_data) = len(data)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # picks either training or validation set
    ix = torch.randint(len(data) - block_size, (batch_size,)) # randomly chooses the batch_size starting position index
    x = torch.stack([data[i:i+block_size] for i in ix]) # x = the input sequence of length block_size (8 tokens)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y = the target sequence (same length), but shifted one position to the right
    return x, y


@torch.no_grad() # Tells PyTorch that we are not going to call .backward() on this function, so it doesn't need to keep track of gradients
# This makes it very memory efficient
def estimate_loss(): # Averages the loss over multiple batches
    out = {}
    model.eval() # Setting the model to evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Setting the model to training phase
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# This specific example is a lot more random
# A bigram language model is a simplified model where you look at the previous single token to predict the next one
# Looks at pairs of consecutive terms (bigram)
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size): # initialization
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # Lookup table:
        # Input: token index (0, 1, 2, 3...)
        # Output: a vector of size vocab_size
        # Eg) In tensor, if you pick 24 character then it will pluck out the 24th row

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # Batch x Time x Channel (Vocab size), in this case, 4x8x65
            logits = logits.view(B*T, C) # Scores for the next characters in the sequence
            targets = targets.view(B*T) # Next characters in the training text
            loss = F.cross_entropy(logits, targets) # Measures how well the program predicted the true next character
            # We are expecting the loss to be loss = -ln(1/65)

        return logits, loss

    
    def generate(self, idx, max_new_tokens): # Takes BxT and makes it +1, +2, +3, etc.
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C), pluck out the last element in the time dimension
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C), ask to give one sample from the distribution
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device) # The lookup table is now on the GPU, so everything becomes a lot faster
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# .optim contains different optimization algorithms
# AdamW is an optimizer that adapts the learning rate for each parameter
# m.parameters() is the model BigramLanguageModel, where .parameters() gives all trainable weights (so the lookup table)
# The optimizer needs to know which parameters to update during the training

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Create the context on the device
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
