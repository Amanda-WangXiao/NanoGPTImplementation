"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"Length of dataset in characters: {len(data):,}","\n")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("All the unique characters:", ''.join(chars),"\n")
print(f"Vocab size: {vocab_size:,}","\n")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("Check stoi items:")
for ch,i in stoi.items():
    print(ch, i)
print("\n")

# create the train and test splits
n = len(data)
print(f"Total Length: {n}")
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print(f"Length of train_data: {len(train_data)}")
print(f"Length of val_data: {len(val_data)}")
print("\n")

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print("\n")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16) # np.uint16's numerical range : 0 ~ 65,535
val_ids = np.array(val_ids, dtype=np.uint16) # np.uint16 saves memory

# tofile() write the data directly to the hard drive in sequence as a raw binary byte stream
# It contains no file headers, shape information, or data type metadata
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin')) # For memory mapping later

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


