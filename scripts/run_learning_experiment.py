"""Validation-tuned ATC versus tabular adaptive k. Test is opened only after locking."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
from pathlib import Path
from statistics import fmean
from time import perf_counter

from mas_jssp.agents.adaptive_atc import K_GRID, TabularKAgent, run_episode
from mas_jssp.learning_data import load_cases, make_split_manifest, scenario_hash, validate_splits
from mas_jssp.learning_statistics import compare_policies


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    paths = sorted(Path("src/mas_jssp").rglob("*.py")) + [Path(__file__)]
    return {str(p.resolve()): digest(p) for p in paths}


def prepare(args):
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory; existing studies are never overwritten")
    if args.runs < 2 or args.test_seeds < 2 or args.episodes < 1 or args.checkpoint_every < 1:
        raise ValueError("Need >=2 runs/test seeds and positive episode/checkpoint counts")
    if any(not 0 < a <= 1 for a in args.alphas) or len(set(args.alphas)) != len(args.alphas):
        raise ValueError("Distinct alphas in (0,1] required")
    manifest = read_json(args.data / "manifest.json")
    available = {item["name"] for item in manifest["instances"]}
    if not set(args.instances) <= available:
        raise ValueError("Some instances are missing from the dataset manifest")
    splits = make_split_manifest(args.instances, args.train_seeds, args.val_seeds, args.test_seeds)
    validate_splits(splits)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", {
        "version": 1, "data": str(args.data.resolve()), "instances": args.instances,
        "dataset_manifest_sha256": digest(args.data / "manifest.json"),
        "jsplib_revision": manifest["revision"], "splits": splits,
        "actions": K_GRID, "alphas": args.alphas, "episodes": args.episodes,
        "checkpoint_every": args.checkpoint_every,
        "training_seeds": [101 + i for i in range(args.runs)],
        "epsilon": {"start": 0.3, "end": 0.05},
        "objective": "mean total_tardiness / total_processing across cases",
        "baseline_selection": "validation only; ties prefer smaller k",
        "policy_selection": "best validation checkpoint per run incl. epoch 0; "
                            "choose ONE alpha by mean validation loss over training seeds",
        "gamma": 1.0, "python": platform.python_version(),
        "source_sha256": source_hashes(),
        "test_protocol": "30 seed blocks by default; average over fixed instances and trained "
                         "policies before Wilcoxon; crossed bootstrap for mean differences",
    })
    print("Protocol saved. No test scenarios evaluated.", flush=True)


def check_protocol(out):
    protocol = read_json(out / "protocol.json")
    validate_splits(protocol["splits"])
    if digest(Path(protocol["data"]) / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest has changed")
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("Source has changed since protocol was frozen; start a new study")
    return protocol


def validation_loss(cases, agent=None, fixed_k=None):
    return fmean(run_episode(jobs, agent=agent, fixed_k=fixed_k)["normalized_tardiness"]
                 for _, jobs in cases)


def train(out):
    if (out / "test_started.json").exists() or (out / "selection.json").exists():
        raise FileExistsError("Study already locked/tested; training cannot be repeated in place")
    protocol = check_protocol(out)
    # Deliberately never load protocol['splits']['test'] in this phase.
    training = load_cases(protocol["data"], protocol["splits"]["train"])
    validation = load_cases(protocol["data"], protocol["splits"]["validation"])
    hashes = {scenario_hash(jobs) for _, jobs in training}
    if hashes & {scenario_hash(jobs) for _, jobs in validation}:
        raise ValueError("Identical train and validation scenarios")
    grid = []
    for k in protocol["actions"]:
        loss = validation_loss(validation, fixed_k=k)
        grid.append({"k": k, "validation_normalized_tardiness": loss})
        print(f"Validation grid k={k}: {loss:.6f}", flush=True)
    write_rows(out / "validation_grid.csv", grid)
    best = min(grid, key=lambda row: (row["validation_normalized_tardiness"], row["k"]))
    best_k, baseline_loss = best["k"], best["validation_normalized_tardiness"]
    print(f"Fixed ATC locked at validation k={best_k}", flush=True)
    models = out / "models"
    models.mkdir(exist_ok=True)
    history, candidates = [], []
    for alpha in protocol["alphas"]:
        for seed in protocol["training_seeds"]:
            agent = TabularKAgent(best_k, alpha, seed, protocol["actions"])
            order_rng = random.Random(seed)
            best_agent, best_loss, best_episode = copy.deepcopy(agent), baseline_loss, 0
            history.append({"alpha": alpha, "training_seed": seed, "episode": 0,
                            "validation_loss": baseline_loss, "train_recent_loss": None,
                            "states": 0})
            recent = []
            order = []
            for episode in range(1, protocol["episodes"] + 1):
                if not order:
                    order = list(range(len(training)))
                    order_rng.shuffle(order)
                _, jobs = training[order.pop()]
                progress = (episode - 1) / max(1, protocol["episodes"] - 1)
                eps = 0.3 + (0.05 - 0.3) * progress
                result = run_episode(jobs, agent=agent, training=True, epsilon=eps)
                recent.append(result["normalized_tardiness"])
                if episode % protocol["checkpoint_every"] == 0 or episode == protocol["episodes"]:
                    score = validation_loss(validation, agent=agent)
                    history.append({"alpha": alpha, "training_seed": seed, "episode": episode,
                                    "validation_loss": score,
                                    "train_recent_loss": fmean(recent), "states": len(agent.q)})
                    recent = []
                    if score < best_loss - 1e-12:
                        best_agent, best_loss, best_episode = copy.deepcopy(agent), score, episode
                    print(f"alpha={alpha} seed={seed} ep={episode} val={score:.6f} "
                          f"best_ep={best_episode}", flush=True)
            name = f"q_alpha{alpha}_seed{seed}.json"
            last_name = f"q_alpha{alpha}_seed{seed}_last.json"
            agent.save(models / last_name)
            best_agent.save(models / name)
            candidates.append({"alpha": alpha, "training_seed": seed, "model": f"models/{name}",
                               "selected_episode": best_episode, "validation_loss": best_loss,
                               "last_trained_model": f"models/{last_name}",
                               "model_sha256": digest(models / name)})
            write_rows(out / "training_history.csv", history)
    means = {a: fmean(c["validation_loss"] for c in candidates if c["alpha"] == a)
             for a in protocol["alphas"]}
    alpha = min(means, key=lambda a: (means[a], a))
    selected = [c for c in candidates if c["alpha"] == alpha]
    write_json(out / "selection.json", {
        "protocol_sha256": digest(out / "protocol.json"),
        "fixed_k": best_k, "baseline_validation_loss": baseline_loss,
        "selected_alpha": alpha, "alpha_validation_means": means,
        "models": selected, "all_candidates": candidates,
        "locked_before_test": True,
    })
    print(f"Selection locked: k={best_k}, alpha={alpha}; test has not been opened.", flush=True)


def test(out):
    if (out / "test_summary.json").exists():
        raise FileExistsError("Final test already exists; inspect it instead of retuning")
    protocol = check_protocol(out)
    selection = read_json(out / "selection.json")
    if digest(out / "protocol.json") != selection["protocol_sha256"]:
        raise ValueError("Protocol changed after model selection")
    agents = []
    for model in selection["models"]:
        path = out / model["model"]
        if digest(path) != model["model_sha256"]:
            raise ValueError("Selected model has changed")
        agents.append((model, TabularKAgent.load(path)))
    write_json(out / "test_started.json", {
        "selection_sha256": digest(out / "selection.json"),
        "protocol_sha256": digest(out / "protocol.json"),
    })
    cases = load_cases(protocol["data"], protocol["splits"]["test"])
    rows = []
    for index, (case, jobs) in enumerate(cases, 1):
        variants = [("ATC_TUNED", None, None)] + [
            ("Q_ATC", model, agent) for model, agent in agents]
        for method, model, agent in variants:
            start = perf_counter()
            result = run_episode(jobs, agent=agent,
                                 fixed_k=selection["fixed_k"] if agent is None else None)
            rows.append({
                **case, "scenario_sha256": scenario_hash(jobs), "method": method,
                "training_seed": None if model is None else model["training_seed"],
                "selected_episode": None if model is None else model["selected_episode"],
                **result, "runtime_seconds": perf_counter() - start,
                "k_counts": json.dumps(result["k_counts"], sort_keys=True),
            })
        if index % len(protocol["instances"]) == 0:
            print(f"Test: {index}/{len(cases)} scenarios complete", flush=True)
    for model, agent in agents:
        if agent.payload() != TabularKAgent.load(out / model["model"]).payload():
            raise AssertionError("Test mutated a policy")
    write_rows(out / "test_runs.csv", rows)
    seeds = sorted({c["seed"] for c, _ in cases})
    instances = protocol["instances"]
    lookup = {(r["method"], r["training_seed"], r["seed"], r["instance"]): r for r in rows}
    comparisons = {}
    for metric in ("normalized_tardiness", "total_tardiness", "makespan"):
        baseline = [[lookup[("ATC_TUNED", None, seed, name)][metric]
                     for name in instances] for seed in seeds]
        values = [[[lookup[("Q_ATC", model["training_seed"], seed, name)][metric]
                    for name in instances] for seed in seeds] for model, _ in agents]
        comparisons[metric] = compare_policies(values, baseline)
    # Only normalized tardiness is the prespecified confirmatory test.
    for metric in ("total_tardiness", "makespan"):
        for key in ("wilcoxon_statistic", "wilcoxon_p_two_sided", "wilcoxon_pairs",
                    "zero_difference_blocks"):
            comparisons[metric].pop(key)
    summary = {
        "selection_sha256": digest(out / "selection.json"),
        "fixed_k": selection["fixed_k"], "selected_alpha": selection["selected_alpha"],
        "models": selection["models"], "test_scenarios": len(cases),
        "test_schedule_runs": len(rows), "independent_test_seed_blocks": len(seeds),
        "training_runs": len(agents), "comparisons": comparisons,
        "test_unchanged_models": True,
    }
    write_json(out / "test_summary.json", summary)
    result = comparisons["normalized_tardiness"]
    verdict = ("На тесте среднее опоздание агента ниже."
               if result["paired_mean_delta"] < 0 else
               "На тесте среднее опоздание агента выше."
               if result["paired_mean_delta"] > 0 else
               "Средние результаты на тесте совпали.")
    lines = [
        "# Табличный агент выбора k: результат закрытого теста", "",
        f"Фиксированный ATC: k={selection['fixed_k']}, выбран только на validation.",
        f"Выбранная alpha={selection['selected_alpha']}. "
        f"Независимых обучений: {len(agents)}; тестовых сценариев: {len(cases)}.",
        f"Всего тестовых расписаний: {len(rows)}; все проверены валидатором.", "",
        "| Метрика | Настроенный ATC | Q-агент |", "|---|---:|---:|",
    ]
    for metric, r in comparisons.items():
        lines.append(f"| {metric} | {r['baseline_mean']:.6f} | {r['agent_mean']:.6f} |")
    lines += ["", verdict,
              f"Изменение отношения средних: {result['relative_mean_change_percent']}%.",
              f"95% bootstrap CI разности Q − ATC: {result['paired_mean_delta_ci95']}.",
              f"SD средних по независимым обучениям: "
              f"{result['agent_std_across_training_run_means']:.6f}.",
              f"Двусторонний Wilcoxon: p={result['wilcoxon_p_two_sided']:.6g}, "
              f"{result['wilcoxon_pairs']} независимых блоков seed.",
              f"Победы / равенства / проигрыши по блокам: "
              f"{result['wins_ties_losses_by_seed_block']}.", "",
              "## Выбор моделей по validation", "",
              "| Seed обучения | Выбранный эпизод | Validation loss |",
              "|---|---:|---:|"]
    for m in selection["models"]:
        lines.append(f"| {m['training_seed']} | {m['selected_episode']} | "
                     f"{m['validation_loss']:.6f} |")
    lines += ["", "Эпизод 0 означает, что validation выбрала исходную фиксированную политику; "
              "это не успех обучения.", "", "## Границы вывода", "",
              "- Один обучаемый координатор задаёт k всем аукционам текущего события.",
              "- Агрегированное состояние частично наблюдаемо, строгая марковость не доказана.",
              "- Train/validation/test имеют разные seed поступлений и сроков; "
              "маршруты задач общие. Перенос на новые маршруты/размеры не проверен.",
              "- Сравнение условно на фиксированной панели экземпляров JSPLIB.",
              "- Bootstrap пересэмплирует независимо запуски обучения и блоки seed, "
              "сохраняя панель экземпляров. При пяти обучениях интервалы предварительные.",
              "- Wilcoxon использует одну пару на блок seed после усреднения по моделям "
              "и экземплярам; предполагает симметричность распределения разностей.",
              "- Статистическая проверка предзадана только для normalized_tardiness; "
              "makespan и сырое опоздание — описательные показатели.",
              "- Результат не доказывает преимущества или бесполезности Deep RL.",
              "", "Подробности: docs/adaptive_atc.md. Сырые результаты: test_runs.csv.",
              "Выбор параметров: validation_grid.csv, training_history.csv, selection.json.",
              "Протокол и контрольные суммы: protocol.json, test_started.json."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(verdict, flush=True)
    print(f"Test report saved to {out / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "prepare", "train", "test"), default="all")
    parser.add_argument("--data", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_atc"))
    parser.add_argument("--instances", nargs="+", default=["ft06", "ft10", "ft20", "la01", "la16"])
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--val-seeds", type=int, default=10)
    parser.add_argument("--test-seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.3])
    args = parser.parse_args()
    if args.phase in ("all", "prepare"):
        prepare(args)
    if args.phase in ("all", "train"):
        train(args.output)
    if args.phase in ("all", "test"):
        test(args.output)


if __name__ == "__main__":
    main()
