import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import AgentMetrics, save_metrics, load_metrics


def test_record_iteration_accumulates():
    m = AgentMetrics(mode="test")
    m.record_iteration(1.0, {"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 0})
    m.record_iteration(0.5, {"prompt_tokens": 150, "completion_tokens": 30, "cached_tokens": 100})
    assert m.iterations == 2
    assert m.prompt_tokens == 250
    assert m.completion_tokens == 50
    assert m.cached_tokens == 100
    assert m.cache_hits == 1


def test_hit_rate_and_cache_ratio():
    m = AgentMetrics(mode="test")
    m.record_iteration(1.0, {"prompt_tokens": 100, "completion_tokens": 0, "cached_tokens": 0})
    m.record_iteration(1.0, {"prompt_tokens": 100, "completion_tokens": 0, "cached_tokens": 100})
    assert m.hit_rate == 50.0
    assert m.cache_ratio == 50.0


def test_zero_iterations_no_division_error():
    m = AgentMetrics(mode="test")
    assert m.hit_rate == 0.0
    assert m.cache_ratio == 0.0
    assert m.summary()  # should not raise


def test_save_and_load_roundtrip(tmp_path):
    m = AgentMetrics(mode="roundtrip_test")
    m.record_iteration(1.2, {"prompt_tokens": 500, "completion_tokens": 40, "cached_tokens": 300})
    m.total_time = 3.5

    out_path = str(tmp_path / "result.json")
    saved_path = save_metrics(m, out_path)
    assert saved_path == out_path

    loaded = load_metrics(out_path)
    assert loaded["mode"] == "roundtrip_test"
    assert loaded["prompt_tokens"] == 500
    assert loaded["cached_tokens"] == 300
    assert loaded["total_time"] == 3.5