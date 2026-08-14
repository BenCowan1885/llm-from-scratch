# LLM From Scratch

A character-level GPT-style language model, built from scratch in PyTorch, following
Andrej Karpathy's "Let's build GPT" as a guide. Built as a hands-on introduction to
transformer architecture ahead of starting a PhD in managing and monitoring AI models.

## Goal

Not to build a useful or high-performing model — the goal was to understand, at the
code level, how every component of a transformer works, and to build Python fluency
and Git/GitHub competence along the way.

## What it does

Trains a small transformer on the Tiny Shakespeare dataset (~1.1M characters) and
generates Shakespeare-style text, one character at a time.

## Architecture

- Character-level tokenization (65-character vocabulary)
- Token + positional embeddings
- [n_layer] stacked transformer blocks, each with:
  - Multi-head self-attention ([n_head] heads, masked/causal)
  - Feedforward layer with ReLU
  - Residual connections and layer normalization
- Dropout regularization (0.2)
- Trained with AdamW, on Apple M2 GPU via PyTorch MPS

## Final results

- Final train loss: 1.5859
- Final val loss: val loss 1.7557
- Sample output: 
Font: I bus.

PARDY:
Coll; old be so I'll that and it be'er we.

LEONTY:

Ay, ten Lason:
Which plit

## Project structure

- `gpt.py` — the final, working model: data loading, architecture, training, generation
- `exploration.py` — earlier exploratory/scratch code from building up the attention
  mechanism step by step (kept for reference, not part of the final model)
- `input.txt` — training data (Tiny Shakespeare)

## What I learned

## 1. Recap of the comprehensions/lambdas/f-strings syntax

**`{ ch:i for i,ch in enumerate(chars) }`** — a dictionary comprehension.
- `enumerate(chars)` pairs each character in your `chars` list with its index: `(0, 'a'), (1, 'b'), ...`
- `for i, ch in enumerate(chars)` loops through those pairs, unpacking each into `i` (the number) and `ch` (the character).
- `{ ch: i for ... }` builds a dictionary using `ch` as the key and `i` as the value for every pair.
- Result: `stoi`, a dictionary mapping each character to its integer index (e.g. `{'a': 12, 'b': 13, ...}`).

**`encode = lambda s: [stoi[c] for c in s]`** — a lambda (anonymous, one-line function).
- `lambda s: ...` defines a function taking one argument, `s`, without needing a full `def` block.
- `[stoi[c] for c in s]` is a list comprehension: for every character `c` in the input string `s`, look up its integer via the `stoi` dictionary, and collect all those integers into a list.
- Result: calling `encode("hi")` returns something like `[46, 47]` — the string converted into a list of integers.

**`f"step {iter}: loss {loss.item():.4f}"`** — an f-string (formatted string literal).
- The `f` before the opening quote tells Python to look inside the string for `{...}` placeholders and substitute in actual values.
- `{iter}` inserts the current value of the variable `iter` directly into the string.
- `{loss.item():.4f}` inserts the value of `loss.item()`, formatted using `.4f` — meaning "fixed-point notation, 4 digits after the decimal point" — so instead of printing something like `2.1928431...`, it prints `2.1928`.
- Result: a clean, readable line like `step 1000: loss 2.1928`, built from live variable values without needing separate `print` arguments or manual string concatenation.

## 2. README summary paragraphs

This project was my first hands-on introduction to how large language models actually work, built from scratch in PyTorch by following Andrej Karpathy's "Let's build GPT" as a guide, ahead of starting a PhD in managing and monitoring AI models. Starting with limited Python experience beyond a mathematical background, I built up a full character-level transformer step by step: a tokenizer, a simple bigram baseline model to establish why longer context matters, and then a real self-attention mechanism — Query/Key/Value projections, causal masking, and scaled dot-product attention — which I extended into multi-head attention, feedforward layers, residual connections, and layer normalization, before stacking several of these into a complete transformer architecture. The final model was scaled up and trained on my MacBook Air's GPU (via PyTorch's MPS backend), bringing validation loss down from an untrained ~4.3 to around 1.75, and producing text with genuinely Shakespeare-like structure — character names, punctuation, and sentence rhythm — despite still being a small, toy-scale model.

Beyond the architecture itself, this project meaningfully improved my Python fluency — comprehensions, lambda functions, f-strings, classes and inheritance via `nn.Module`, and general tensor manipulation in PyTorch — largely through debugging real, self-introduced bugs rather than just reading about correct code. A sign error in a masking operation (`inf` vs `-inf`) taught me exactly why that mechanism works the way it does; a misplaced `=`/`==` and an indentation slip clarified how strictly Python's syntax and whitespace are interpreted; and a silently broken line inside my `generate()` function — which produced no error at all, just quietly useless output — was a good lesson in methodical debugging rather than assuming code matches what I remember writing.

I also used this project to learn Git and GitHub from complete scratch, committing at each architectural milestone, which left me with both a working knowledge of version control and a readable history of how the project actually evolved. More broadly, building this by hand — rather than only reading about transformers — surfaced practical failure modes (masking bugs, GPU/CPU device mismatches, silent logic errors) that theory alone wouldn't have shown me, which feels like a genuinely useful starting point heading into a PhD focused on monitoring and managing AI systems in practice.

## Requirements

See `requirements.txt`. Built and run with Python 3.13 in a virtual environment.

## Acknowledgements

Based on Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
video and [nanoGPT](https://github.com/karpathy/nanoGPT).