"""
Context Window & Token Budget Manager
Stage 01 - Challenge 02
"""


class TokenBudgetManager:
    GLOBAL_BOS_OVERHEAD = 1        # 1 token for <bos>
    MESSAGE_FRAMING_OVERHEAD = 3   # 3 tokens per message for <start_turn>, role, <end_turn>

    def __init__(
        self,
        max_context_tokens: int,
        reserved_completion_tokens: int,
        special_tokens: dict[str, int] | None = None,
    ):
        """
        Initializes the TokenBudgetManager.
        
        Args:
            max_context_tokens: Total context window size supported by model.
            reserved_completion_tokens: Max output tokens reserved for generation.
            special_tokens: Mapping of special tokens to IDs (e.g. {'<bos>': 1, '<eos>': 2, '<pad>': 0, '<unk>': 3}).
        """
        self.max_context_tokens = max_context_tokens
        self.reserved_completion_tokens = reserved_completion_tokens
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

    def count_tokens(self, text: str) -> int:
        """
        Estimates the number of tokens in a text string.
        Uses a whitespace and subword approximation: each whitespace-separated word counts as 1 token.
        Returns 0 for empty or whitespace-only text.
        
        Args:
            text: Input string.
            
        Returns:
            Number of tokens.
        """
        # TODO: Implement this function
        # 1. Strip whitespace.
        # 2. If text is empty, return 0.
        # 3. Otherwise, split on whitespace and return the count of words/tokens.
        raise NotImplementedError("Implement count_tokens")

    def calculate_available_budget(self, current_prompt_tokens: int) -> int:
        """
        Calculates remaining available prompt token capacity.
        
        Available Budget = max(0, (max_context_tokens - reserved_completion_tokens) - current_prompt_tokens)
        
        Args:
            current_prompt_tokens: Number of tokens currently consumed by prompt.
            
        Returns:
            Remaining token budget available for prompt expansion (minimum 0).
        """
        # TODO: Implement this function
        # 1. Compute max_allowed_prompt_tokens = self.max_context_tokens - self.reserved_completion_tokens
        # 2. Return max(0, max_allowed_prompt_tokens - current_prompt_tokens)
        raise NotImplementedError("Implement calculate_available_budget")

    def format_and_truncate_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], int]:
        """
        Enforces token budget limits on a list of chat messages:
        - Calculates token cost per message: count_tokens(content) + MESSAGE_FRAMING_OVERHEAD (3 tokens).
        - Adds GLOBAL_BOS_OVERHEAD (1 token) to the entire conversation.
        - Preserves 'system' messages intact (pins them at the beginning).
        - If total tokens exceed max allowed prompt tokens, drops oldest non-system messages (FIFO)
          until total token cost fits within budget.
        
        Args:
            messages: List of message dictionaries, each having 'role' and 'content' keys.
                      e.g. [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
                      
        Returns:
            Tuple of (truncated_messages, total_token_count).
        """
        # TODO: Implement this function
        # 1. Calculate max_allowed_prompt = self.max_context_tokens - self.reserved_completion_tokens
        # 2. Separate system messages from non-system (chat) messages.
        # 3. Define helper to compute total tokens:
        #    total = GLOBAL_BOS_OVERHEAD + sum(count_tokens(m["content"]) + MESSAGE_FRAMING_OVERHEAD for m in active_msgs)
        # 4. While chat_msgs is not empty and total tokens > max_allowed_prompt:
        #    chat_msgs.pop(0)  # Drop oldest non-system message
        # 5. Return (system_msgs + chat_msgs, total_tokens)
        raise NotImplementedError("Implement format_and_truncate_messages")
