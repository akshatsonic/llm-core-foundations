import unittest
from tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=25)

    def test_get_stats(self):
        vocab = {
            ('l', 'o', 'w', '</w>'): 5,
            ('l', 'o', 'w', 'e', 'r', '</w>'): 2,
            ('n', 'e', 'w', 'e', 's', 't', '</w>'): 6,
            ('w', 'i', 'd', 'e', 's', 't', '</w>'): 3,
        }
        stats = self.tokenizer.get_stats(vocab)
        
        # 'e' + 's' appears in newest (6) and widest (3) -> 9
        self.assertEqual(stats.get(('e', 's')), 9)
        # 's' + 't' appears in newest (6) and widest (3) -> 9
        self.assertEqual(stats.get(('s', 't')), 9)
        # 'l' + 'o' appears in low (5) and lower (2) -> 7
        self.assertEqual(stats.get(('l', 'o')), 7)
        # 'o' + 'w' appears in low (5) and lower (2) -> 7
        self.assertEqual(stats.get(('o', 'w')), 7)
        # 'w' + '</w>' appears in low (5) -> 5
        self.assertEqual(stats.get(('w', '</w>')), 5)

    def test_merge_vocab(self):
        vocab = {
            ('n', 'e', 'w', 'e', 's', 't', '</w>'): 6,
            ('w', 'i', 'd', 'e', 's', 't', '</w>'): 3,
        }
        merged = self.tokenizer.merge_vocab(('e', 's'), vocab)
        expected = {
            ('n', 'e', 'w', 'es', 't', '</w>'): 6,
            ('w', 'i', 'd', 'es', 't', '</w>'): 3,
        }
        self.assertEqual(merged, expected)

    def test_train_and_vocab_size(self):
        corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
        tok = BPETokenizer(vocab_size=20)
        tok.train(corpus)

        self.assertGreater(len(tok.merges), 0)
        self.assertLessEqual(len(tok.token_to_id), 20)
        self.assertIn("</w>", tok.token_to_id)
        self.assertIn("<unk>", tok.token_to_id)
        # Check bidirectional consistency
        for token, tid in tok.token_to_id.items():
            self.assertEqual(tok.id_to_token[tid], token)

    def test_encode_and_decode(self):
        corpus = "hug hug pug pug pun bun bun bun"
        tok = BPETokenizer(vocab_size=20)
        tok.train(corpus)

        encoded = tok.encode("hug pug bun")
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(x, int) for x in encoded))

        decoded = tok.decode(encoded)
        self.assertEqual(decoded, "hug pug bun")

    def test_roundtrip_preservation(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        tok = BPETokenizer(vocab_size=40)
        tok.train(corpus)

        test_sentences = [
            "the quick brown fox",
            "jumps over lazy dog",
            "the brown dog jumps",
        ]
        for sent in test_sentences:
            encoded = tok.encode(sent)
            decoded = tok.decode(encoded)
            self.assertEqual(decoded, sent)


if __name__ == "__main__":
    unittest.main()
