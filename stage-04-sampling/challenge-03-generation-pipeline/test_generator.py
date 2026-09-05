"""
Unit tests for Challenge 03: End-to-End Autoregressive Generation Pipeline
"""

import unittest
from typing import Iterator

from generator import TextGenerationPipeline


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "The": 3,
            " cat": 4,
            " sat": 5,
            " on": 6,
            " the": 7,
            " mat": 8,
            " STOP": 9,
            "\n": 10,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        # Simple rule-based mock encoder
        tokens = []
        words = text.split(" ")
        for i, w in enumerate(words):
            if not w:
                continue
            key = w if i == 0 else f" {w}"
            if key in self.vocab:
                tokens.append(self.vocab[key])
            elif w in self.vocab:
                tokens.append(self.vocab[w])
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.inv_vocab.get(tid, "") for tid in token_ids)


class MockStoryModel:
    """
    Deterministic mock model that generates a predictable story:
    "The" (3) -> " cat" (4) -> " sat" (5) -> " on" (6) -> " the" (7) -> " mat" (8) -> "<eos>" (2)
    """

    def __init__(self):
        self.transition = {
            3: 4,  # The -> cat
            4: 5,  # cat -> sat
            5: 6,  # sat -> on
            6: 7,  # on -> the
            7: 8,  # the -> mat
            8: 2,  # mat -> <eos>
        }
        self.vocab_size = 12

    def forward(self, input_ids: list[int]) -> list[list[float]]:
        last_token = input_ids[-1] if input_ids else 3
        next_token = self.transition.get(last_token, 2)  # default to <eos>

        # Return logits where target next_token has logit 10.0, others have 0.0
        logits = [0.0] * self.vocab_size
        logits[next_token] = 10.0
        return [logits]  # [seq_len, vocab_size]


class MockLoopingModel:
    """Model that keeps outputting " cat" forever to test max_new_tokens and stop_tokens."""

    def __init__(self):
        self.vocab_size = 12

    def forward(self, input_ids: list[int]) -> list[list[float]]:
        # If last token is 5, output " STOP" (9), else output " cat" (4)
        last_token = input_ids[-1] if input_ids else 3
        next_token = 9 if last_token == 5 else 4

        logits = [0.0] * self.vocab_size
        logits[next_token] = 10.0
        return [logits]


class TestTextGenerationPipeline(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.special_tokens = {"<eos>": 2, "<bos>": 1, "<pad>": 0}

    def test_generate_end_to_end(self):
        model = MockStoryModel()
        pipeline = TextGenerationPipeline(model, self.tokenizer, self.special_tokens)

        result = pipeline.generate(prompt="The", max_new_tokens=10, temperature=0.1)
        self.assertEqual(result, "The cat sat on the mat")

    def test_generate_stream(self):
        model = MockStoryModel()
        pipeline = TextGenerationPipeline(model, self.tokenizer, self.special_tokens)

        stream = pipeline.generate_stream(prompt="The", max_new_tokens=10, temperature=0.1)
        self.assertTrue(isinstance(stream, Iterator))

        chunks = list(stream)
        self.assertEqual(chunks, [" cat", " sat", " on", " the", " mat"])

    def test_stop_on_eos_token(self):
        model = MockStoryModel()
        pipeline = TextGenerationPipeline(model, self.tokenizer, self.special_tokens)

        # Model produces <eos> after 5 tokens, max_new_tokens is 50
        chunks = list(pipeline.generate_stream(prompt="The", max_new_tokens=50, temperature=0.1))
        self.assertEqual(len(chunks), 5)
        self.assertNotIn("<eos>", chunks)

    def test_max_new_tokens_budget(self):
        model = MockLoopingModel()
        pipeline = TextGenerationPipeline(model, self.tokenizer, self.special_tokens)

        chunks = list(pipeline.generate_stream(prompt="The", max_new_tokens=4, temperature=0.1))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks, [" cat", " cat", " cat", " cat"])

    def test_stop_tokens(self):
        model = MockLoopingModel()
        pipeline = TextGenerationPipeline(model, self.tokenizer, self.special_tokens)

        # Prompt with token 5 (" sat") will trigger model to emit " STOP" (9)
        chunks = list(
            pipeline.generate_stream(
                prompt="The cat sat",
                max_new_tokens=10,
                temperature=0.1,
                stop_tokens=["STOP", " STOP"],
            )
        )
        # Should stop before or upon encountering stop token
        self.assertNotIn(" STOP", chunks)


if __name__ == "__main__":
    unittest.main()
