"""Benchmark nine policies on public instances and paired synthetic scenarios."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
from collections import defaultdict
from pathlib import Path

from mas_jssp.experiments import METHODS, evaluate, make_scenario, verify_controls
from mas_jssp.utils.loaders import load_jssp_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--k", type=float, default=2.0)
    args = parser.parse_args()
    if args.seeds < 1 or args.k <= 0:
        parser.error("seeds and k must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    static = []
    for item in manifest["instances"]:
        path = args.data / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Data checksum mismatch: {path}")
        base = load_jssp_instance(path)
        scenarios = [("static", 0, 0)] + [("due_dates", f, 0) for f in (1.5, 2.0, 3.0)]
        scenarios += [("arrivals", f, seed) for f in (1.5, 2.0) for seed in range(args.seeds)]
        for scenario, factor, seed in scenarios:
            jobs = make_scenario(base, scenario, factor or 2.0, seed)
            paired = []
            for method in METHODS:
                row, _ = evaluate(jobs, method, args.k)
                optimum = item.get("optimum")
                # Reference optima apply ONLY to unchanged static makespan instances.
                row.update(instance=item["name"], jobs=item["jobs"], machines=item["machines"],
                           scenario=scenario, due_factor=factor, seed=seed, atc_k=args.k,
                           static_optimum=optimum if scenario == "static" else None,
                           gap_percent=(100 * (row["makespan"] / optimum - 1)
                                        if scenario == "static" and optimum else None))
                paired.append(row)
            verify_controls(paired)
            rows.extend(paired)
            if scenario == "static":
                static.extend(paired)
        print(f"{item['name']}: {item['jobs']}x{item['machines']}, "
              f"{len(scenarios) * len(METHODS)} valid runs", flush=True)
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_paths = sorted(Path("src/mas_jssp").rglob("*.py")) + [
        Path("scripts/run_benchmarks.py"), Path("scripts/download_benchmarks.py")]
    config = {
        "python": platform.python_version(), "platform": platform.platform(),
        "jsplib_revision": manifest["revision"], "seeds": list(range(args.seeds)),
        "atc_k": args.k, "methods": METHODS,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "note": "Timing is one wall-clock sample per run; no statistical speed claims.",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = [
        "# Результаты экспериментов", "",
        f"Всего {len(rows)} прогонов, все расписания прошли валидацию.",
        "Аукцион и централизованная политика с одинаковыми ставками дали одинаковые "
        "расписания во всех парных проверках.", "",
        "## Статические исходные экземпляры: makespan", "",
        "| Экземпляр | Размер | Известный оптимум | FIFO | SPT | MWKR | Исходный аукцион |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in manifest["instances"]:
        group = {r["method"]: r for r in static if r["instance"] == item["name"]}
        values = " | ".join(f"{group[m]['makespan']:.0f}"
                            for m in ("FIFO", "SPT", "MWKR", "AUCTION_ORIGINAL"))
        report.append(f"| {item['name']} | {item['jobs']}x{item['machines']} | "
                      f"{item.get('optimum') or 'не указан'} | {values} |")
    report += ["", "Без дедлайнов ATC_LOCAL сводится к SPT, исходная ставка — к MWKR. "
               "Опоздание для исходных статических задач обозначено N/A в CSV.",
               "", "## Сценарии со сроками: сравнение с FIFO", "",
               "Ниже среднее относительное изменение суммарного опоздания в паре с FIFO. "
               "Отрицательное значение означает улучшение. Пары с нулевым опозданием "
               "FIFO исключены из относительного показателя и подсчитаны отдельно. "
               "Это описательная статистика, не проверка статистической значимости.",
               "", "| Сценарий | Метод | Среднее изменение, % | Пар | Исключено FIFO=0 |",
               "|---|---|---|---|---|"]
    by_case = defaultdict(dict)
    for row in rows:
        by_case[(row["instance"], row["scenario"], row["due_factor"], row["seed"])][row["method"]] = row
    for scenario in ("due_dates", "arrivals"):
        for method in METHODS:
            deltas = []
            excluded = 0
            for (_, sc, _, _), group in by_case.items():
                if sc != scenario:
                    continue
                reference = group["FIFO"]["total_tardiness"]
                if reference == 0:
                    excluded += 1
                else:
                    deltas.append(100 * (group[method]["total_tardiness"] / reference - 1))
            value = f"{statistics.mean(deltas):.2f}" if deltas else "N/A"
            report.append(f"| {scenario} | {method} | {value} | {len(deltas)} | {excluded} |")
    report += ["", "## Ограничения", "",
               "- JSPLIB — статические данные; сроки и поступления добавлены нами.",
               "- Дедлайн = release_time + due_factor * сумма длительностей заказа.",
               "- arrivals: независимые равномерные поступления на [0, total_work / machines].",
               "- Это конечные партии, не стационарный поток и не реальные данные предприятия.",
               "- ATC_LOCAL: наша явно описанная адаптация; k=2 по умолчанию, веса заказов равны 1.",
               "- Поломки, бюджеты, цены, обучение и протокол Wellman не реализованы.",
               "- Оптимумы и gap применены только к исходным статическим экземплярам.",
               "- Методы детерминированы; seed меняет только сценарий поступлений.",
               "", "Полные измерения: runs.csv. Версии и хеши: config.json. "
               "Методы и статьи: ../../docs/methods_and_sources.md."]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
