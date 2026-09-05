# Challenge 01: Byte-Pair Encoding (BPE) Tokenizer

Welcome to **Stage 01: Tokens & Byte-Pair Encoding (BPE)**!

In this challenge, you will implement a pure-Python Byte-Pair Encoding (BPE) tokenizer from scratch.

---

## 📚 Background: Why Tokens?

Large Language Models (LLMs) do not directly process raw strings or whole words.
- **Raw Characters**: Processing text character-by-character produces very long sequence lengths, increasing quadratic attention costs and making long-range dependencies hard to learn.
- **Whole Words**: Word-level vocabularies suffer from the Out-Of-Vocabulary (OOV) problem (e.g., typos, new slang, morphological variations like *unbelievably*) and create massive, sparse embedding layers.
- **Subword Tokenization (BPE)**: Breaks text into reusable subword units. Frequent words remain single tokens, while rare words are broken down into subword pieces (e.g., `["un", "believ", "able", "ly"]`), resolving the OOV problem with a compact vocabulary.

---

## ⚙️ The BPE Algorithm Step-by-Step

### 1. Representation of Words
During training, we represent a corpus as a frequency dictionary of words, where each word is broken into a tuple of individual characters followed by an end-of-word symbol `</w>` (or space indicator).

Example:
For corpus `"low low low lower newest widest"`:
```python
vocab = {
    ('l', 'o', 'w', '</w>'): 3,
    ('l', 'o', 'w', 'e', 'r', '</w>'): 1,
    ('n', 'e', 'w', 'e', 's', 't', '</w>'): 1,
    ('w', 'i', 'd', 'e', 's', 't', '</w>'): 1,
}
```

### 2. Frequency Counting (`get_stats`)
Count all consecutive symbol pairs across the vocabulary, weighted by each word's frequency:
- Pair `('l', 'o')`: appears 3 times in `"low</w>"` + 1 time in `"lower</w>"` = 4
- Pair `('o', 'w')`: 3 + 1 = 4
- Pair `('e', 's')`: 1 (`"newest"`) + 1 (`"widest"`) = 2
- Pair `('s', 't')`: 1 + 1 = 2

### 3. Finding and Merging the Best Pair (`merge_vocab`)
Find the most frequent pair (e.g., `('e', 's')` or `('l', 'o')`). Merge occurrences of that pair in all word tuples:
If merging `('e', 's')` into `'es'`:
- `('n', 'e', 'w', 'e', 's', 't', '</w>')` $\rightarrow$ `('n', 'e', 'w', 'es', 't', '</w>')`

### 4. Vocabulary & Merge Rules
Record the merged pair in a merge table / list of merges, and add the new subword token to the vocabulary mapping (`token -> id` and `id -> token`). Repeat until the target `vocab_size` is reached or no pairs remain.

### 5. Encoding & Decoding
- **Encode**: Given a new word/text, split into initial character tuples and iteratively apply learned merges in the order they were created. Convert the resulting tokens to integer IDs.
- **Decode**: Given a list of token IDs, lookup their string representations, join them, and remove the end-of-word marker `</w>` (or replace it with whitespace).

---

## 🎯 Task & Passing Criteria

Complete the implementation in [`tokenizer.py`](file:///Users/akshatsonic/.gemini/antigravity/scratch/llm-core-foundations/stage-01-tokens/challenge-01-bpe-tokenizer/tokenizer.py):

1. `get_stats(vocab)`: Accurately count all consecutive symbol pairs weighted by word frequencies.
2. `merge_vocab(pair, vocab)`: Replace consecutive occurrences of `pair = (p0, p1)` in all word tuples with the merged symbol `p0 + p1`.
3. `train(corpus)`: Build initial character vocabulary, iteratively merge the most frequent pairs until `vocab_size` is reached, and record merge rules and token-to-ID mappings.
4. `encode(text)`: Apply merge rules in training order to tokenize unseen text into token IDs.
5. `decode(ids)`: Reconstruct the original text from token IDs, ensuring `decode(encode(text)) == text`.

Run the test suite to verify:
```bash
python3 -m unittest test_tokenizer.py
```

---

## 💡 3-Step Hint Ladder

<details>
<summary><b>Hint 1: get_stats iteration</b></summary>

Iterate over each word tuple and its frequency count in `vocab`. For index `i` from `0` to `len(word) - 2`, increment `pairs[(word[i], word[i+1])] += freq`.
</details>

<details>
<summary><b>Hint 2: merge_vocab sliding replacement</b></summary>

When merging pair `(p0, p1)` in a tuple `word`:
Iterate through `word` with an index `i`. If `i < len(word) - 1` and `word[i] == p0` and `word[i+1] == p1`, append `p0 + p1` to the new word tuple and increment `i` by 2. Otherwise, append `word[i]` and increment `i` by 1.
</details>

<details>
<summary><b>Hint 3: encode using merge list</b></summary>

Split input text by spaces into words. For each word, create a tuple of characters ending with `'</w>'`. For each `pair` in `self.merges` (in creation order), merge that pair across all words in the input. Finally, map each resulting token to `self.token_to_id`.
</details>
