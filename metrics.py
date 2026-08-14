import time
from dataclasses import dataclass, field
import json as _json
from pathlib import Path

@dataclass
class AgentMetrics:
    mode: str
    iterations: int = 0
    ttft_per_iter: list = field(default_factory=list)
    total_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_hits: int = 0

    def record_iteration(self, ttft: float, usage: dict) -> None:
        self.iterations += 1
        self.ttft_per_iter.append(ttft)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        cached = usage.get("cached_tokens", 0)
        self.cached_tokens += cached
        if cached > 0:
            self.cache_hits += 1

    @property
    def hit_rate(self) -> float:
        return 100.0 * self.cache_hits / self.iterations if self.iterations else 0.0

    @property
    def cache_ratio(self) -> float:
        return 100.0 * self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def summary(self) -> str:
        lines = [
            f"Mode: {self.mode}",
            f"Iterations: {self.iterations}",
            f"Total time: {self.total_time:.3f}s",
            f"1st TTFT: {self.ttft_per_iter[0]:.3f}s" if self.ttft_per_iter else "1st TTFT: n/a",
            f"Avg TTFT: {sum(self.ttft_per_iter)/len(self.ttft_per_iter):.3f}s" if self.ttft_per_iter else "Avg TTFT: n/a",
            f"Prompt tokens: {self.prompt_tokens}",
            f"Cached tokens: {self.cached_tokens}",
            f"Cache hit rate: {self.hit_rate:.1f}%",
            f"Cache ratio: {self.cache_ratio:.1f}%",
        ]
        return "\n".join(lines)

def save_metrics(metrics: "AgentMetrics", output_path: str = None) -> str:
    if output_path is None:
        output_path = f"result_{metrics.mode}_{int(time.time())}.json"
    data = {
        "mode": metrics.mode,
        "iterations": metrics.iterations,
        "ttft_per_iter": metrics.ttft_per_iter,
        "total_time": metrics.total_time,
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "cached_tokens": metrics.cached_tokens,
        "cache_hits": metrics.cache_hits,
        "hit_rate": metrics.hit_rate,
        "cache_ratio": metrics.cache_ratio,
    }
    Path(output_path).write_text(_json.dumps(data, indent=2))
    return output_path

def load_metrics(path: str) -> dict:
    return _json.loads(Path(path).read_text())

def print_comparison_table(results: list) -> None:
    header = f"{'Mode':<16}{'Iters':<7}{'1st TTFT':<10}{'Avg TTFT':<10}{'Total(s)':<10}{'Prompt':<9}{'Cached':<9}{'Hit%':<7}{'Cache%':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        first_ttft = r["ttft_per_iter"][0] if r["ttft_per_iter"] else 0
        avg_ttft = sum(r["ttft_per_iter"]) / len(r["ttft_per_iter"]) if r["ttft_per_iter"] else 0
        print(
            f"{r['mode']:<16}{r['iterations']:<7}{first_ttft:<10.3f}{avg_ttft:<10.3f}"
            f"{r['total_time']:<10.3f}{r['prompt_tokens']:<9}{r['cached_tokens']:<9}"
            f"{r['hit_rate']:<7.1f}{r['cache_ratio']:<8.1f}"
        )