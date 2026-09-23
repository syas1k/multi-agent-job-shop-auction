"""Paired statistics; never flatten repeated policies into independent cases."""
import numpy as np
from scipy.stats import wilcoxon


def compare_policies(agent_values, baseline_values, bootstrap_seed=7401, samples=5000):
    """Agent shape [training_run, scenario_seed, instance], baseline [seed, instance].

    Benchmark instances are a fixed panel. Resample independent training runs and
    scenario-seed blocks separately (crossed bootstrap), retaining all instances.
    """
    agents = np.asarray(agent_values, dtype=float)
    baseline = np.asarray(baseline_values, dtype=float)
    if agents.ndim != 3 or baseline.shape != agents.shape[1:]:
        raise ValueError("Expected [run, seed, instance] and [seed, instance]")
    if agents.shape[0] < 2 or agents.shape[1] < 2 or samples < 1:
        raise ValueError("Need at least two runs and scenario seeds")
    if not np.isfinite(agents).all() or not np.isfinite(baseline).all():
        raise ValueError("Non-finite metrics")
    differences = agents - baseline[None, :, :]
    run_means = agents.mean(axis=(1, 2))
    # One independent pair per scenario-seed block, NOT per (model, case).
    paired = np.round(differences.mean(axis=(0, 2)), decimals=12)
    if np.all(paired == 0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(paired, zero_method="pratt", alternative="two-sided", method="auto")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    rng = np.random.default_rng(bootstrap_seed)
    boot_agent, boot_delta, boot_baseline = [], [], []
    nr, ns, ni = agents.shape
    for _ in range(samples):
        rr = rng.integers(nr, size=nr)
        ss = rng.integers(ns, size=ns)
        sampled = agents[rr[:, None], ss[None, :], :]
        sampled_base = baseline[ss, :]
        boot_agent.append(float(sampled.mean()))
        boot_baseline.append(float(sampled_base.mean()))
        boot_delta.append(float((sampled - sampled_base[None, :, :]).mean()))
    ci = lambda values: np.quantile(values, [0.025, 0.975]).tolist()
    return {
        "agent_mean": float(agents.mean()),
        "baseline_mean": float(baseline.mean()),
        "agent_std_across_training_run_means": float(run_means.std(ddof=1)),
        "agent_std_across_scenario_seed_means": float(agents.mean(axis=(0, 2)).std(ddof=1)),
        "baseline_std_across_scenario_seed_means": float(baseline.mean(axis=1).std(ddof=1)),
        "agent_mean_ci95": ci(boot_agent), "baseline_mean_ci95": ci(boot_baseline),
        "paired_mean_delta": float(differences.mean()),
        "paired_mean_delta_ci95": ci(boot_delta),
        "per_run_means": run_means.tolist(),
        "relative_mean_change_percent": (
            100 * float(differences.mean()) / float(baseline.mean())
            if baseline.mean() else None),
        "wilcoxon_statistic": statistic, "wilcoxon_p_two_sided": pvalue,
        "wilcoxon_pairs": ns, "zero_difference_blocks": int(sum(paired == 0)),
        "wins_ties_losses_by_seed_block": [
            int(sum(paired < 0)), int(sum(paired == 0)), int(sum(paired > 0))],
        "bootstrap_samples": samples, "bootstrap_seed": bootstrap_seed,
        "note": "Fixed instance panel; crossed run/seed-block bootstrap. Wilcoxon "
                "conditions on these trained policies and assumes symmetric paired differences.",
    }
