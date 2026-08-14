# KV Cache Demonstration (LangChain/langchain + OpenAI)

A ReAct-style agent with local filesystem tools that demonstrates how OpenAI's
prompt caching (KV cache) behaves under six implementation patterns one
correct, five common anti-patterns. Small-looking context changes can silently
kill cache hits and hurt latency/cost.

Built as a LangChain/OpenAI reimplementation of a Moonshot Kimi-based reference
project, same six modes, same idea: OpenAI automatically caches prompts over
~1024 tokens and reports `cached_tokens` in `usage.prompt_tokens_details`.

## What is KV Cache?

The model's attention layer stores key-value pairs for tokens it has already
processed. If a later prompt starts with an identical, unchanged prefix, the
provider can reuse those cached values instead of recomputing them cutting
latency and cost. Change anything in that prefix (even reordering identical
content) and the cache misses.

## Modes

| Mode | What it does | Effect |
|---|---|---|
| `correct` | Fixed system prompt, fixed tool order, messages list built once and appended to | Cache builds up normally |
| `dynamic_system` | Injects a timestamp into the system prompt every call | Full prefix invalidation every call |
| `shuffled_tools` | Randomizes tool definition order every call | Breaks the prefix even though content is identical |
| `dynamic_profile` | Appends a changing "user profile" block after the stable history | Only invalidates from that point forward earlier prefix can still cache |
| `sliding_window` | Keeps only the last N messages | Looks efficient but breaks the aligned/growing prefix; can also cause the model to lose earlier context entirely |
| `text_format` | Flattens the whole conversation into one plain-text blob each call | Breaks the API's structured-message caching entirely |

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Usage

```bash
# Interactive menu
python main.py

# Single mode
python main.py --mode correct

# Run all 6 modes and print a comparison table
python main.py --compare

# Offline: rebuild the comparison table from saved result_*.json files (no API key)
python main.py --report
python main.py --report --input "result_correct_*.json" "result_text_format_*.json"

# Custom task / root directory
python main.py --mode correct --task "Summarize all .py files" --root-dir ../other-project
```

## Metrics

Each run tracks: iterations, TTFT per iteration, total time, prompt/completion
tokens, cached tokens, cache hit rate (% of iterations with any cache), and
cache ratio (% of prompt tokens served from cache). Results are saved to
`result_<mode>_<timestamp>.json` for later offline comparison.

## Example findings (5-file, ~1.8K-token task)
Mode Iters 1st TTFT Avg TTFT Total(s) Prompt Cached Hit% Cache%

correct 3 1.605 1.956 5.900 1844 1280 33.3 69.4
dynamic_system 3 0.750 2.596 7.798 1904 0 0.0 0.0
shuffled_tools 3 0.849 1.814 5.481 1844 0 0.0 0.0
dynamic_profile 3 0.762 1.774 5.333 1895 1280 33.3 67.5
sliding_window 3 0.876 2.048 6.161 1727 1152 33.3 66.7
text_format 4 0.689 1.368 5.482 1966 0 0.0 0.0

**Key finding  position matters more than "dynamic content" per se.**
`dynamic_system` (rewrites the prefix) and `dynamic_profile` (appends after
the stable prefix) both inject changing data every call, but only one destroys
the cache. Corrupting the *start* of the prompt breaks everything after it;
appending stays compatible with caching for the untouched portion.

**Caveat vs. the Kimi reference:** OpenAI's cache is not perfectly
deterministic  `shuffled_tools` and `sliding_window` scored differently
across separate runs (sometimes 0%, sometimes ~65%+). Treat single-run
numbers as noisy; average over a few runs for a fair comparison.

## Architecture
kv-cache-langchain/
├── agent.py # 6 agent classes, one per mode
├── tools.py # sandboxed read_file / find / grep (LangChain @tool)
├── metrics.py # AgentMetrics, save/load, comparison table
├── main.py # CLI: --mode / --compare / --report / interactive menu
├── sample_data/ # generated test files the agent reads
├── tests/ # offline pytest checks (no API key needed)
├── requirements.txt
└── result_*.json # saved run metrics (gitignored)

## Tests

```bash
pip install pytest
python -m pytest tests -v
```

## Troubleshooting

- **Always 0 cached tokens**: your prompt is likely under OpenAI's ~1024-token
  caching threshold for every individual call  check per-iteration
  `prompt_tokens` in the console output, not just the summed total.
- **`BadRequestError` about `tool` role messages**: happened with naive
  sliding-window truncation cutting an assistant's `tool_calls` message while
  keeping the paired `tool` response  fixed by extending the window backward
  to the enclosing assistant message (see `_safe_window()` in `agent.py`).
- **Numbers vary between runs**: OpenAI's prompt caching is not fully
  deterministic; rerun a few times before drawing conclusions from one pass.
