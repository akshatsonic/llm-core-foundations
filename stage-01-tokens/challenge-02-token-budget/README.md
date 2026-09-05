# Challenge 02: LLM Context Window & Token Budget Manager

Welcome to **Stage 01: Challenge 02**!

In this challenge, you will implement a **Token Budget Manager** that simulates how backend LLM systems (e.g. Anthropic, OpenAI, Gemini APIs) enforce context window limits, manage special framing tokens, reserve completion capacity, and intelligently truncate chat histories.

---

## 📚 Background: Context Windows & Request Budgeting

Every Large Language Model has a finite **Context Window** ($N_{\text{max}}$ tokens), representing the combined total of the input prompt tokens and the generated output tokens.

$$\text{Context Window} = \text{Input Prompt Tokens} + \text{Generated Completion Tokens}$$

When designing LLM backend applications, you must budget token allocations before dispatching a request:
1. **Reserved Completion Tokens (`max_tokens`)**: The guaranteed token budget reserved for the model to generate its response.
2. **System Prompt**: Core instructions, personality, tools, and safety constraints that should almost **never** be dropped.
3. **Framing & Special Tokens**: Control tokens (`<bos>`, `<eos>`, `<start_turn>`, `<end_turn>`, role headers) that delineate speaker turns in chat format (e.g. ChatML or Llama-style templates).
4. **Message History**: The conversational turns (`user`, `assistant`). As history grows, it quickly exceeds the context window and must be truncated via FIFO (First-In, First-Out) sliding window while preserving system instructions and the latest user turn.

---

## ⚙️ Budgeting & Message Framing Rules

### 1. Token Budget Equation
$$\text{Max Allowed Prompt Tokens} = \text{max\_context\_tokens} - \text{reserved\_completion\_tokens}$$

- If `current_prompt_tokens > Max Allowed Prompt Tokens`, available budget is `0` (or negative).
- `calculate_available_budget(current_prompt_tokens)` returns:
  $$\max(0, (\text{max\_context\_tokens} - \text{reserved\_completion\_tokens}) - \text{current\_prompt\_tokens})$$

### 2. Message Framing & Token Counting
For each message dictionary `{"role": "...", "content": "..."}`:
- Each message incurs a fixed turn framing overhead of **3 tokens** (e.g., `<start_turn>`, `role` tag, and `<end_turn>`).
- Content tokens are counted via `count_tokens(content)`.
- Total tokens for a single message = `count_tokens(message["content"]) + 3`.
- There is also a global conversation prefix overhead of **1 token** (`<bos>`).

Thus, for a list of messages:
$$\text{Total Tokens} = 1 + \sum_{m \in \text{messages}} (\text{count\_tokens}(m[\text{"content"}]) + 3)$$

### 3. Truncation Strategy
When formatting a multi-turn chat list:
1. Separate `system` message (if present) from conversational messages (`user`, `assistant`). System message is pinned and must **never** be dropped.
2. Calculate total token requirement. If total tokens exceed `Max Allowed Prompt Tokens`:
   - Drop the **oldest** non-system messages one-by-one (FIFO) until the remaining messages (plus system message + global BOS overhead) fit within the budget.
3. Return the truncated list of messages along with the exact total token count.

---

## 🎯 Task & Passing Criteria

Complete the implementation in [`token_budget.py`](file:///Users/akshatsonic/.gemini/antigravity/scratch/llm-core-foundations/stage-01-tokens/challenge-02-token-budget/token_budget.py):

1. `count_tokens(text)`: Returns the token count for a text string using whitespace and subword estimation.
2. `calculate_available_budget(current_prompt_tokens)`: Calculates remaining token capacity for prompt expansion.
3. `format_and_truncate_messages(messages)`: Intelligently truncates conversation history, preserves system instructions, accounts for framing overhead, and returns `(truncated_messages, total_tokens)`.

Run the test suite to verify:
```bash
python3 -m unittest test_token_budget.py
```

---

## 💡 3-Step Hint Ladder

<details>
<summary><b>Hint 1: Message cost calculation</b></summary>

Write a helper method `message_tokens(msg)`:
`return self.count_tokens(msg.get("content", "")) + self.MESSAGE_FRAMING_OVERHEAD` (where framing overhead is 3).
Global total is `self.GLOBAL_BOS_OVERHEAD` (1) + sum of all message tokens.
</details>

<details>
<summary><b>Hint 2: Separating system vs non-system turns</b></summary>

Filter out the system message (e.g., `system_msgs = [m for m in messages if m["role"] == "system"]`) and non-system messages (`chat_msgs = [m for m in messages if m["role"] != "system"]`).
</details>

<details>
<summary><b>Hint 3: FIFO Truncation Loop</b></summary>

Start with all `chat_msgs`. While `chat_msgs` is not empty and the total token count (`GLOBAL_BOS_OVERHEAD + sum(message_tokens(m) for m in (system_msgs + chat_msgs))`) exceeds `max_allowed_prompt_tokens`, remove the oldest turn: `chat_msgs.pop(0)`. Finally, combine `system_msgs + chat_msgs`.
</details>
