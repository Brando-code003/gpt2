# GPT-2 from Scratch in NumPy

A learning project: building the pieces of GPT-2 by hand, with only NumPy, to understand how a language model really works inside.

## What is GPT-2, simply?

GPT-2 is a model that **guesses the next word**. You give it some text, and it gives every possible next word a score. Pick a word, add it to the text, and repeat: that is how it writes.

It does this in 4 steps:

1. **Tokenize:** turn text into numbers (token ids). `"I love"` becomes `[40, 1842]`.
2. **Embed:** turn each id into a vector, a list of numbers that describes the word, plus a vector for its position in the sentence.
3. **Transformer blocks:** each word looks at the words before it (**attention**) to understand context. In "the river bank", `bank` looks at `river` to know it means a riverside, not a money bank.
4. **Predict:** turn each word's final vector into a score for every word in the vocabulary (50,257 words for GPT-2). The highest scores are the most likely next words.

## Files

| File | What it is for |
|---|---|
| `tokenizer.py` | Turns text into token ids with GPT-2's real tokenizer (`tiktoken`). |
| `learning_gpt2.ipynb` | Step-by-step notebook with diagrams: self-attention, trainable Q/K/V, the causal mask, multi-head attention (slow and fast), layer norm, feed-forward and the transformer block. Start here to learn. |
| `main.py` | The whole model in one file: embeddings, transformer blocks, the GPT class, the loss, and a `generate` function that writes text. |

## Setup

You need Python 3 and a few packages:

```bash
pip install numpy tiktoken matplotlib jupyter
```

## How to run

**Tokenizer:** prints the token ids for a sentence.

```bash
python3 tokenizer.py
```

**Notebook:** opens it in your browser; run the cells from top to bottom.

```bash
jupyter notebook learning_gpt2.ipynb
```

**Full model:** runs the model on a sentence, prints the loss, then generates a few words.

```bash
python3 main.py
```

## What to expect

The model is **not trained yet**: its weights are random numbers. So:

- The **loss** is about 10.8, the same as guessing randomly among 50,257 words.
- The **generated text** is nonsense, like `I love Brando revealing, clothes Rifle`.

That is normal. Training (the backward pass and gradient descent) is the next step, and it is what teaches the model to write real sentences.
