import unittest
from token_budget import TokenBudgetManager


class TestTokenBudgetManager(unittest.TestCase):
    def setUp(self):
        self.manager = TokenBudgetManager(
            max_context_tokens=100,
            reserved_completion_tokens=20,
            special_tokens={"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
        )

    def test_count_tokens(self):
        self.assertEqual(self.manager.count_tokens(""), 0)
        self.assertEqual(self.manager.count_tokens("   "), 0)
        self.assertEqual(self.manager.count_tokens("hello world"), 2)
        self.assertEqual(self.manager.count_tokens("You are a helpful AI assistant."), 6)

    def test_calculate_available_budget(self):
        # Max allowed prompt = 100 - 20 = 80
        self.assertEqual(self.manager.calculate_available_budget(30), 50)
        self.assertEqual(self.manager.calculate_available_budget(80), 0)
        # Should not go below 0 when exceeding max allowed
        self.assertEqual(self.manager.calculate_available_budget(95), 0)

    def test_message_framing_overhead(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},  # 5 tokens + 3 framing = 8
            {"role": "user", "content": "Hello there!"},                    # 2 tokens + 3 framing = 5
        ]
        # Total = 1 (BOS) + 8 + 5 = 14 tokens
        truncated_msgs, token_count = self.manager.format_and_truncate_messages(messages)
        self.assertEqual(len(truncated_msgs), 2)
        self.assertEqual(token_count, 14)

    def test_system_prompt_preservation_and_fifo_truncation(self):
        # Create budget with strict max allowed prompt = 50 - 10 = 40 tokens
        strict_mgr = TokenBudgetManager(max_context_tokens=50, reserved_completion_tokens=10)
        
        # System prompt: 5 tokens + 3 framing = 8
        # Message 1 (oldest user): 10 tokens + 3 framing = 13
        # Message 2 (assistant): 10 tokens + 3 framing = 13
        # Message 3 (newest user): 10 tokens + 3 framing = 13
        # Total with all: 1 (BOS) + 8 + 13 + 13 + 13 = 48 tokens > 40 limit.
        # Dropping Message 1: 1 (BOS) + 8 + 13 + 13 = 35 tokens <= 40 limit.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "word " * 10},
            {"role": "assistant", "content": "word " * 10},
            {"role": "user", "content": "word " * 10},
        ]
        
        truncated_msgs, total_tokens = strict_mgr.format_and_truncate_messages(messages)
        
        # System message preserved + 2 newest messages kept
        self.assertEqual(len(truncated_msgs), 3)
        self.assertEqual(truncated_msgs[0]["role"], "system")
        self.assertEqual(truncated_msgs[1]["role"], "assistant")
        self.assertEqual(truncated_msgs[2]["role"], "user")
        self.assertEqual(total_tokens, 35)

    def test_extreme_overflow_preserves_system_prompt(self):
        # Budget max prompt = 30 - 10 = 20 tokens
        strict_mgr = TokenBudgetManager(max_context_tokens=30, reserved_completion_tokens=10)
        
        # System: 2 tokens + 3 framing = 5
        # User 1: 20 tokens + 3 framing = 23 -> exceeds 20 on its own!
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "word " * 20},
        ]
        
        truncated_msgs, total_tokens = strict_mgr.format_and_truncate_messages(messages)
        # Even if user message must be dropped to fit, system message remains
        self.assertEqual(truncated_msgs[0]["role"], "system")
        self.assertEqual(len(truncated_msgs), 1)
        # BOS (1) + 5 = 6 tokens
        self.assertEqual(total_tokens, 6)


if __name__ == "__main__":
    unittest.main()
