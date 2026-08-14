import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

from tools import ALL_TOOLS, set_root_dir
from metrics import AgentMetrics
from datetime import datetime
import random

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a ReAct-style coding assistant with access to filesystem tools "
    "(read_file, find, grep). Use them to answer the user's task, then give a "
    "final answer. Keep tool use minimal and purposeful."
)

# OpenAI-format tool schemas (built once, order fixed = cache-friendly)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a text file, given a path relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find files matching a glob pattern relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a text pattern inside files under the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

class KVCacheAgentCorrect:
    """Stable-context ReAct agent: fixed system prompt, fixed tool order,
    messages list built once and appended to (never rebuilt) -> cache-friendly."""

    def __init__(self, root_dir: str = ".", max_iterations: int = 6):
        self.client = OpenAI()
        set_root_dir(root_dir)
        self.max_iterations = max_iterations
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, task: str) -> AgentMetrics:
        metrics = AgentMetrics(mode="correct")
        start = time.time()

        self.messages.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            iter_start = time.time()
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
            )
            ttft = time.time() - iter_start

            usage = response.usage
            cached = 0
            if usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

            metrics.record_iteration(ttft, {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": cached,
            })

            msg = response.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                metrics.total_time = time.time() - start
                print(f"\nFinal answer:\n{msg.content}")
                return metrics

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_fn = TOOLS_BY_NAME.get(fn_name)
                result = tool_fn.invoke(args) if tool_fn else f"Error: unknown tool {fn_name}"
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        metrics.total_time = time.time() - start
        return metrics

class KVCacheAgentDynamicSystem:
    """Anti-pattern: injects a timestamp into the system prompt every request,
    so the prefix changes on every call -> cache invalidation."""

    def __init__(self, root_dir: str = ".", max_iterations: int = 6):
        self.client = OpenAI()
        set_root_dir(root_dir)
        self.max_iterations = max_iterations
        # NOTE: no fixed system message stored here it's rebuilt every call.
        self.history = []  # user/assistant/tool turns only, system excluded

    def _build_system_message(self) -> dict:
        return {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nCurrent time: {datetime.now().isoformat()}",
        }

    def run(self, task: str) -> AgentMetrics:
        metrics = AgentMetrics(mode="dynamic_system")
        start = time.time()

        self.history.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            # Rebuild the full messages list from scratch each iteration,
            # with a fresh system prompt -> breaks the stable prefix.
            messages = [self._build_system_message()] + self.history

            iter_start = time.time()
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            ttft = time.time() - iter_start

            usage = response.usage
            cached = 0
            if usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

            metrics.record_iteration(ttft, {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": cached,
            })

            msg = response.choices[0].message
            self.history.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                metrics.total_time = time.time() - start
                print(f"\nFinal answer:\n{msg.content}")
                return metrics

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_fn = TOOLS_BY_NAME.get(fn_name)
                result = tool_fn.invoke(args) if tool_fn else f"Error: unknown tool {fn_name}"
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        metrics.total_time = time.time() - start
        return metrics

class KVCacheAgentShuffledTools:
    """Anti-pattern: randomizes tool definition order every request.
    Same tools, same system prompt, same messages -- but reordering the
    tools list breaks the prefix since it changes what precedes the rest
    of the prompt."""

    def __init__(self, root_dir: str = ".", max_iterations: int = 6):
        self.client = OpenAI()
        set_root_dir(root_dir)
        self.max_iterations = max_iterations
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _shuffled_tool_schemas(self) -> list:
        shuffled = TOOL_SCHEMAS.copy()
        random.shuffle(shuffled)
        return shuffled

    def run(self, task: str) -> AgentMetrics:
        metrics = AgentMetrics(mode="shuffled_tools")
        start = time.time()

        self.messages.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            iter_start = time.time()
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self._shuffled_tool_schemas(),  # reordered every call
            )
            ttft = time.time() - iter_start

            usage = response.usage
            cached = 0
            if usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

            metrics.record_iteration(ttft, {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": cached,
            })

            msg = response.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                metrics.total_time = time.time() - start
                print(f"\nFinal answer:\n{msg.content}")
                return metrics

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_fn = TOOLS_BY_NAME.get(fn_name)
                result = tool_fn.invoke(args) if tool_fn else f"Error: unknown tool {fn_name}"
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        metrics.total_time = time.time() - start
        return metrics

if __name__ == "__main__":
    task = "Find all .py files and summarize what tools.py does in 2 sentences."

    print("=== CORRECT ===")
    agent1 = KVCacheAgentCorrect(root_dir=".")
    m1 = agent1.run(task)
    print("\n" + m1.summary())

    print("\n=== DYNAMIC_SYSTEM ===")
    agent2 = KVCacheAgentDynamicSystem(root_dir=".")
    m2 = agent2.run(task)
    print("\n" + m2.summary())

    print("\n=== SHUFFLED_TOOLS ===")
    agent3 = KVCacheAgentShuffledTools(root_dir=".")
    m3 = agent3.run(task)
    print("\n" + m3.summary())