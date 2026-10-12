# NanoGPT Implementation  

## Introduction
A minimal GPT implementation in PyTorch. It trains a character-level language model on Tiny Shakespeare and samples new text from the checkpoint. The stack follows the GPT-2 design: token and position embeddings, causal self-attention, and a feed-forward block, with AdamW.  

## Common Commands(For Mac M chips)
python data/shakespeare_char/prepare.py  
python train.py --device=mps --compile=False --dtype=float32
python train.py --device=mps --compile=False --dtype=float32 --init_from=resume
python train.py --device=mps --compile=False --dtype=float32 --max_iters=2000 --lr_decay_iters=2000 --eval_interval=100
python sample.py --device=mps
python sample.py --device=mps --start="ROMEO:" --num_samples=3 --max_new_tokens=300

## Original Git Repo
I am deeply grateful to the original author for contributing his code; my development is based on his repository. Please refer to the link below to visit the original author's repository:
link[https://github.com/karpathy/nanoGPT]
