import argparse
import glob as globmod
import sys

from agent import (
    KVCacheAgentCorrect,
    KVCacheAgentDynamicSystem,
    KVCacheAgentShuffledTools,
    KVCacheAgentDynamicProfile,
    KVCacheAgentSlidingWindow,
    KVCacheAgentTextFormat,
)
from metrics import save_metrics, load_metrics, print_comparison_table

MODES = {
    "correct": KVCacheAgentCorrect,
    "dynamic_system": KVCacheAgentDynamicSystem,
    "shuffled_tools": KVCacheAgentShuffledTools,
    "dynamic_profile": KVCacheAgentDynamicProfile,
    "sliding_window": KVCacheAgentSlidingWindow,
    "text_format": KVCacheAgentTextFormat,
}

DEFAULT_TASK = (
    "Read all files in sample_data/ (there are 5: module_0.py through module_4.py). "
    "For each one, note what the process_data function and DataProcessor class do. "
    "Then give a combined summary in 4-5 sentences."
)

def run_single(mode: str, task: str, root_dir: str, output: str = None) -> dict:
    agent_cls = MODES[mode]
    agent = agent_cls(root_dir=root_dir)
    print(f"\n=== {mode.upper()} ===")
    metrics = agent.run(task)
    print("\n" + metrics.summary())
    path = save_metrics(metrics, output)
    print(f"\nSaved: {path}")
    return load_metrics(path)

def run_compare(task: str, root_dir: str) -> None:
    results = []
    for mode in MODES:
        result = run_single(mode, task, root_dir)
        results.append(result)
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print_comparison_table(results)

def run_report(input_patterns: list) -> None:
    if not input_patterns:
        input_patterns = ["result_*.json"]
    files = []
    for pattern in input_patterns:
        files.extend(globmod.glob(pattern))
    files = sorted(set(files))
    if not files:
        print("No result files found.")
        return
    results = [load_metrics(f) for f in files]
    print_comparison_table(results)

def interactive_menu(task: str, root_dir: str) -> None:
    mode_list = list(MODES.keys())
    while True:
        print("\nModes:")
        for i, m in enumerate(mode_list, 1):
            print(f"  {i}. {m}")
        print(f"  {len(mode_list) + 1}. Compare All")
        print("  0. Exit")
        choice = input("> ").strip()
        if choice == "0":
            break
        elif choice == str(len(mode_list) + 1):
            run_compare(task, root_dir)
        elif choice.isdigit() and 1 <= int(choice) <= len(mode_list):
            run_single(mode_list[int(choice) - 1], task, root_dir)
        else:
            print("Invalid choice.")

def main():
    parser = argparse.ArgumentParser(description="KV Cache Demonstration (LangChain/OpenAI)")
    parser.add_argument("--mode", choices=list(MODES.keys()), help="Run a single mode")
    parser.add_argument("--compare", action="store_true", help="Run all modes and compare")
    parser.add_argument("--report", action="store_true", help="Offline report from saved JSON")
    parser.add_argument("--input", nargs="*", help="Glob patterns for --report")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Custom task for the agent")
    parser.add_argument("--root-dir", default=".", help="Root dir for filesystem tools")
    parser.add_argument("--output", default=None, help="Output JSON path (single mode only)")
    parser.add_argument("--no-interactive", action="store_true", help="Skip menu if no flags given")
    args = parser.parse_args()

    if args.report:
        run_report(args.input)
    elif args.compare:
        run_compare(args.task, args.root_dir)
    elif args.mode:
        run_single(args.mode, args.task, args.root_dir, args.output)
    elif args.no_interactive:
        print("No mode specified.")
        sys.exit(1)
    else:
        interactive_menu(args.task, args.root_dir)

if __name__ == "__main__":
    main()