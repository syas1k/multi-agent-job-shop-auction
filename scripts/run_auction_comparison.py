"""Сравнить простые правила, новые торги и CP-SAT на одинаковых задачах.

Пример:
python scripts/run_auction_comparison.py data/benchmarks/ft06.txt --output results/new_pilot
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from statistics import fmean

from mas_jssp.agents.budget_auction import run_budget_auction
from mas_jssp.agents.local_rule_auction import run_local_rules
from mas_jssp.auction.protocols import PROTOCOLS
from mas_jssp.cp_scheduler import quantize_jobs, run_cp_offline, run_cp_online
from mas_jssp.experiments import evaluate
from mas_jssp.learning_data import generate_scenario, scenario_hash
from mas_jssp.utils.loaders import load_jssp_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=420000)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--objective", choices=("tardiness", "makespan"), default="tardiness")
    parser.add_argument("--cp-seconds", type=float, default=0.05)
    parser.add_argument("--offline-seconds", type=float, default=1.0)
    parser.add_argument("--credits", type=float, default=1.0)
    parser.add_argument("--price-step", type=float, default=0.1)
    parser.add_argument("--max-rounds", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a NEW output directory; archived results cannot be overwritten")
    if args.seeds < 1 or args.scale < 1 or args.max_rounds < 1:
        parser.error("Counts and scale must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.cp_seconds, args.offline_seconds, args.price_step)):
        parser.error("Time limits and price step must be positive")
    if not math.isfinite(args.credits) or args.credits < 0:
        parser.error("Credits must be nonnegative")
    if len({p.resolve() for p in args.instances}) != len(args.instances) or len(
            {p.stem for p in args.instances}) != len(args.instances):
        parser.error("Duplicate inputs or instance names")
    # Проверка установки до создания папки: если библиотеки нет, результатов не будет.
    from ortools.sat.python import cp_model  # noqa: F401
    bases = [(p, load_jssp_instance(p)) for p in args.instances]
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    # Хешируем только относящиеся к этой версии исходники, без отложенного прототипа.
    omitted = {"autonomous_auction.py", "local_neural.py"}
    sources = [p for p in sorted((root / "src" / "mas_jssp").rglob("*.py"))
               if p.name not in omitted] + [Path(__file__).resolve()]
    config = {
        "purpose": "Exploratory comparison; not validation-selected or confirmatory",
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("ortools", "numpy")},
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items() if k != "instances"},
        "instances": [{"name": p.stem, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p, _ in bases],
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sources},
        "time_conversion": "All methods receive identical integer ticks, floor(x*scale+0.5). "
                           "Reported time metrics are divided by scale; hashes use ticks.",
        "information": "CP_OFFLINE knows future arrivals; all other methods are online.",
        "bidding": "Fixed common allowance/urgency policy, not equilibrium or learned bidding.",
        "credits": "Winner-only payments are removed, not transferred to machines; no replenishment.",
    }
    # На данном Windows применена локальная подписанная DLL; фиксируем её хеш.
    # Путь не зашит в алгоритм: на другом ПК достаточно актуального VC++ Runtime.
    local_runtime = Path(sys.prefix) / "DLLs" / "msvcp140.dll"
    if local_runtime.exists():
        config["local_msvc_runtime_sha256"] = hashlib.sha256(local_runtime.read_bytes()).hexdigest()
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, traces = [], {}
    for path, base in bases:
        for seed in range(args.seed_base, args.seed_base + args.seeds):
            # Это один сценарий. Его копии получают ВСЕ сравниваемые методы.
            jobs = quantize_jobs(generate_scenario(base, path.stem, seed), args.scale)
            common = {"instance": path.stem, "seed": seed,
                      "scenario_sha256": scenario_hash(jobs)}
            first = path == bases[0][0] and seed == args.seed_base
            tasks = [(method, lambda m=method: evaluate(jobs, m, 2.0)[0])
                     for method in ("FIFO", "EDD", "SPT", "MWKR", "ATC_LOCAL")]
            tasks += [("LOCAL_RULES", lambda: run_local_rules(jobs))]
            tasks += [(protocol, lambda p=protocol: run_budget_auction(
                jobs, p, args.credits, args.price_step, args.max_rounds, first))
                for protocol in PROTOCOLS]
            tasks += [
                ("CP_ONLINE", lambda: run_cp_online(
                    jobs, args.objective, args.cp_seconds, 0, first)),
                ("CP_OFFLINE", lambda: run_cp_offline(
                    jobs, args.objective, args.offline_seconds, 0)),
            ]
            for method, run in tasks:
                started = time.perf_counter()
                result = run()
                elapsed = time.perf_counter() - started
                trace = result.pop("trace", None)
                if trace and first:
                    traces[method] = trace
                if result.get("valid"):
                    work = sum(o.duration for j in jobs for o in j.operations)
                    result["normalized_tardiness"] = result["total_tardiness"] / work
                    # Возвращаем показатели из тактов в исходные единицы времени.
                    for key in ("makespan", "total_tardiness", "objective",
                                "lower_bound", "mean_flow_time"):
                        if result.get(key) is not None:
                            result[key] /= args.scale
                rows.append({**common, **result, "method": method,
                             "runtime_seconds": elapsed})
                # Сразу сохраняем каждое измерение. Даже при сбое следующего метода
                # уже выполненные запуски останутся в журнале, а не пропадут из памяти.
                with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(rows[-1]) + "\n")
                print(f"  {path.stem} {seed}: {method}, valid={result.get('valid')}, "
                      f"runtime={elapsed:.2f}s", flush=True)
            print(f"Completed {path.stem}, scenario {seed}", flush=True)

    summary = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        all_rows = [r for r in rows if r["method"] == method]
        valid = [r for r in all_rows if r.get("valid")]
        # Неудачные CP-запуски не скрываются. Если есть сбой, общего среднего нет.
        complete = len(valid) == len(all_rows)
        summary[method] = {
            "runs": len(all_rows), "valid_runs": len(valid),
            "all_valid": complete,
            **{metric: fmean(r[metric] for r in valid) if complete else None
               for metric in ("makespan", "total_tardiness", "normalized_tardiness")},
            "mean_runtime_seconds": fmean(r["runtime_seconds"] for r in all_rows),
            "total_rounds": sum(r.get("rounds", 0) for r in all_rows),
            "total_messages": sum(r.get("logical_messages", 0) for r in all_rows),
            "fallback_calls": sum(r.get("fallback_calls", 0) for r in all_rows),
        }
    # Важный контроль: меняют ли платежи ПОЛНОЕ расписание, а не только цифру расходов?
    payment_differences = 0
    for path, _ in bases:
        for seed in range(args.seed_base, args.seed_base + args.seeds):
            pair = {r["method"]: r for r in rows
                    if r["instance"] == path.stem and r["seed"] == seed}
            payment_differences += int(pair["first_price"]["schedule_sha256"] !=
                                       pair["second_price"]["schedule_sha256"])
    for name, payload in (("runs.json", rows), ("summary.json", summary),
                          ("example_traces.json", {"time_unit": "integer ticks",
                                                   "scale": args.scale, "traces": traces})):
        (args.output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Среднее по всей панели может скрывать разницу маленьких и больших задач.
    # Поэтому рядом сохраняем отдельные показатели для каждого экземпляра.
    by_instance = {}
    for path, _ in bases:
        by_instance[path.stem] = {}
        for method in summary:
            own = [r for r in rows if r["instance"] == path.stem and r["method"] == method]
            complete = all(r.get("valid") for r in own)
            by_instance[path.stem][method] = {
                "runs": len(own), "valid_runs": sum(bool(r.get("valid")) for r in own),
                **{key: fmean(r[key] for r in own) if complete else None for key in
                   ("makespan", "total_tardiness", "normalized_tardiness")},
                "mean_runtime_seconds": fmean(r["runtime_seconds"] for r in own),
            }
    (args.output / "by_instance.json").write_text(
        json.dumps(by_instance, indent=2), encoding="utf-8")
    report = [
        "# Пилот: аукционные механизмы и CP-SAT", "",
        "Это разведочный эксперимент без подбора по validation и без проверки значимости.",
        "Все методы получают одну и ту же округлённую задачу. Время в таблице — исходные единицы.",
        "CP_OFFLINE знает будущие заказы: это отдельный ориентир, а не равноправный онлайн-метод.",
        "CP_ONLINE перепланирует только поступившие заказы; при отсутствии решения использует ATC.",
        "", "| Метод | Допустимых | Makespan | Норм. опоздание | Runtime, с |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, s in summary.items():
        span = "N/A" if s["makespan"] is None else f'{s["makespan"]:.4f}'
        tardy = "N/A" if s["normalized_tardiness"] is None else f'{s["normalized_tardiness"]:.6f}'
        report.append(f'| {method} | {s["valid_runs"]}/{s["runs"]} | {span} | {tardy} | '
                      f'{s["mean_runtime_seconds"]:.4f} |')
    report += ["", f"Первая и вторая цена дали разные расписания в {payment_differences} сценариях.",
               "Это эффект заданной политики и модели бюджетов, не доказательство превосходства типа аукциона.",
               "Платежи условные: списываются у победителя, не повышают полезность станка.",
               "Время локальных правил включает диагностический теневой расчёт; точное сравнение скорости требует его уравнивания.",
               "Исходники, настройки, статусы решателя и версии сохранены рядом с отчётом.", ""]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
