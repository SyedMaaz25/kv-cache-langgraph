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

    def __init__(self, root_dir: str = ".", max_iterations: int = 8):
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
            print(f"  [iter] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

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

    def __init__(self, root_dir: str = ".", max_iterations: int = 8):
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
            print(f"  [iter] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

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

    def __init__(self, root_dir: str = ".", max_iterations: int = 8):
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
            print(f"  [iter] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

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

class KVCacheAgentDynamicProfile:
    """Anti-pattern: injects a changing user profile/credits block into the
    prompt every call. Common in production (e.g. showing live usage/balance
    in context) -- looks harmless but invalidates the cached prefix each time."""

    def __init__(self, root_dir: str = ".", max_iterations: int = 8):
        self.client = OpenAI()
        set_root_dir(root_dir)
        self.max_iterations = max_iterations
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._credits = 1000  # simulating live counter

    def _profile_message(self) -> dict:
        # Simulating a counter that changes each call (e.g. usage/credits ticking down)
        self._credits -= random.randint(1, 5)
        return {
            "role": "user",
            "content": f"[User profile: credits_remaining={self._credits}, session_active=true]",
        }

    def run(self, task: str) -> AgentMetrics:
        metrics = AgentMetrics(mode="dynamic_profile")
        start = time.time()

        self.messages.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            # Injecting the changing profile block right before each call
            call_messages = self.messages + [self._profile_message()]

            iter_start = time.time()
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=call_messages,
                tools=TOOL_SCHEMAS,
            )
            ttft = time.time() - iter_start

            usage = response.usage
            cached = 0
            if usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            print(f"  [iter] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

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

class KVCacheAgentSlidingWindow:
    """Anti-pattern: keeps only the last N messages to 'save tokens'.
    Looks efficient but breaks the stable growing prefix the cache relies on."""

    def __init__(self, root_dir: str = ".", max_iterations: int = 8, window_size: int = 5):
        self.client = OpenAI()
        set_root_dir(root_dir)
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.system_message = {"role": "system", "content": SYSTEM_PROMPT}
        self.history = []

    def _safe_window(self) -> list:
        """Take the last window_size messages, but if that slice starts mid-way
        through a tool-call exchange (orphaning a 'tool' message), extend
        backward to include the assistant message that made the calls."""
        window = self.history[-self.window_size:]
        start_idx = len(self.history) - len(window)

        while window and window[0].get("role") == "tool" and start_idx > 0:
            start_idx -= 1
            window = self.history[start_idx:]

        return window

    def run(self, task: str) -> AgentMetrics:
        metrics = AgentMetrics(mode="sliding_window")
        start = time.time()

        self.history.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            #  system message + only the last `window_size` history messages
            windowed = self._safe_window()
            call_messages = [self.system_message] + windowed

            iter_start = time.time()
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=call_messages,
                tools=TOOL_SCHEMAS,
            )
            ttft = time.time() - iter_start

            usage = response.usage
            cached = 0
            if usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            print(f"  [iter] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

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

class KVCacheAgentTextFormat:
    """Anti-pattern: flattens the whole conversation into one plain-text blob
    per call instead of using structured messages -> breaks the API's expected
    format and any prefix caching tied to structured message boundaries."""

    def __init__(self, root_dir: str = ".", max_iterations: int = 8):
        self.client = OpenAI()
        set_root_dir(root_dir)
        self.max_iterations = max_iterations
        self.history = []  # list of (role, content) tuples

    def _render_as_text(self, task: str) -> str:
        lines = [SYSTEM_PROMPT, "", f"User task: {task}", ""]
        for role, content in self.history:
            lines.append(f"[{role.upper()}]: {content}")
        return "\n".join(lines)

    def run(self, task: str) -> AgentMetrics:
        metrics = AgentMetrics(mode="text_format")
        start = time.time()

        for _ in range(self.max_iterations):
            # Rebuiling the ENTIRE conversation as one flat text blob each call
            flat_text = self._render_as_text(task)
            call_messages = [{"role": "user", "content": flat_text}]

            iter_start = time.time()
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=call_messages,
                tools=TOOL_SCHEMAS,
            )
            ttft = time.time() - iter_start

            usage = response.usage
            cached = 0
            if usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            print(f"  [iter] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

            metrics.record_iteration(ttft, {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": cached,
            })

            msg = response.choices[0].message
            self.history.append(("assistant", msg.content or "(tool call)"))

            if not msg.tool_calls:
                metrics.total_time = time.time() - start
                print(f"\nFinal answer:\n{msg.content}")
                return metrics

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_fn = TOOLS_BY_NAME.get(fn_name)
                result = tool_fn.invoke(args) if tool_fn else f"Error: unknown tool {fn_name}"
                self.history.append(("tool_result", f"{fn_name}({args}) -> {result}"))

        metrics.total_time = time.time() - start
        return metrics

if __name__ == "__main__":
    task = (
        "Read all files in sample_data/ (there are 5: module_0.py through module_4.py). "
        "For each one, note what the process_data function and DataProcessor class do. "
        "Then give a combined summary in 4-5 sentences."
    )

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

    print("\n=== DYNAMIC_PROFILE ===")
    agent4 = KVCacheAgentDynamicProfile(root_dir=".")
    m4 = agent4.run(task)
    print("\n" + m4.summary())

    print("\n=== SLIDING_WINDOW ===")
    agent5 = KVCacheAgentSlidingWindow(root_dir=".")
    m5 = agent5.run(task)
    print("\n" + m5.summary())

    print("\n=== TEXT_FORMAT ===")
    agent6 = KVCacheAgentTextFormat(root_dir=".")
    m6 = agent6.run(task)
    print("\n" + m6.summary())