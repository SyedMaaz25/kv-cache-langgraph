import time
from dataclasses import dataclass, field

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