# English to Urdu Neural Machine Translation Transformer

A Transformer model built from scratch in PyTorch for translating English sentences into Urdu, following the architecture from "Attention Is All You Need". Achieves a BLEU score of 0.2974 on the validation set.

## What this project does

This is a full encoder decoder Transformer, implemented without using any pre-built transformer library. It includes:

- Custom tokenizer and vocabulary building for both English and Urdu
- Positional encoding
- Multi head attention (self attention and cross attention)
- Position wise feed forward layers
- Full encoder and decoder stacks
- Greedy decoding and beam search for generating translations
- BLEU score evaluation

## Architecture

- Model dimension: 512
- Encoder and decoder layers: 6 each, as in the original paper
- Attention heads: 8
- Feed forward dimension: 2048
- Dropout: 0.3
- Max sequence length: 100

## Training setup

- Batch size: 64
- Learning rate: 0.0001, with the warmup and decay schedule from the paper (warmup steps: 8000)
- Epochs: 50, with early stopping (patience of 10 epochs)
- Label smoothing: 0.1, to reduce overconfidence and improve generalization
- Gradient clipping at 1.0

## Results

- Validation BLEU score: 0.2974

## Project structure

```
en_ur_transformer.ipynb   Full training and inference pipeline
```

The notebook covers:

1. Imports
2. Tokenizer and vocabulary classes
3. Dataset class for the parallel English Urdu corpus
4. Positional encoding
5. Multi head attention
6. Position wise feed forward network
7. Encoder and decoder layers
8. Full Transformer model
9. Data loading and preprocessing
10. Training configuration and training loop
11. Training and validation loss curves
12. Greedy decoding translation function
13. Beam search translation function
14. BLEU score calculation and evaluation
15. Interactive translation interface
16. Translation quality analysis
17. Model export for inference

## How to use

1. Prepare a parallel corpus as two plain text files, one line per sentence, aligned by line number:
   - An English file (for example `train.en`)
   - An Urdu file (for example `train.ur`)
2. Update the file paths in the data loading cell
3. Run the notebook cells in order to build vocabularies, train the model, and evaluate it
4. After training, translate your own sentences:
   ```python
   translation = translate(model, "Hello, how are you?", src_vocab, tgt_vocab)
   ```
   or use beam search for better quality:
   ```python
   translation = beam_search_translate(model, "Hello, how are you?", src_vocab, tgt_vocab, beam_size=4)
   ```

## Requirements

See `requirements.txt`. Main dependencies are PyTorch, numpy, matplotlib, and tqdm.

## Notes

This was built as an assignment for a Generative AI course, with the goal of implementing the Transformer architecture from scratch rather than using a library like Hugging Face. Vocabulary size, dataset size, and compute budget were limited, so the BLEU score reflects a from scratch student implementation rather than a production translation system.