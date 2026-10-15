"""CP-SAT: поиск расписания целиком и перепланирование по поступившим заказам.

CP-SAT — существующий решатель OR-Tools. Мы задаём ему переменные, ограничения
и цель. Это централизованный контрольный метод, а не аукцион или нейросеть.
"""
import copy
import math
from collections import Counter

from .environment.environment import Environment, MachineState
from .environment.priority import priority
from .experiments import schedule_fingerprint
from .metrics.metrics import makespan, total_tardiness
from .metrics.validate import validate_schedule


def quantize_jobs(jobs, scale=10):
    """Одинаково перевести время ВСЕХ методов в целые такты перед сравнением."""
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("Positive integer time scale required")
    copied = copy.deepcopy(jobs)
    for job in copied:
        # Округление до ближайшего такта: floor(x + 0.5).
        # Это новая, явно указанная точность задачи, а не исходные дробные времена.
        if not math.isfinite(job.release_time) or job.release_time < 0:
            raise ValueError("Invalid release")
        job.release_time = math.floor(job.release_time * scale + 0.5)
        if job.due_date is not None:
            if not math.isfinite(job.due_date):
                raise ValueError("Invalid deadline")
            job.due_date = math.floor(job.due_date * scale + 0.5)
        for op in job.operations:
            if not math.isfinite(op.duration) or op.duration <= 0:
                raise ValueError("Invalid duration")
            if op.start is not None or op.end is not None:
                raise ValueError("Expected unscheduled jobs")
            op.duration = max(1, math.floor(op.duration * scale + 0.5))
    return copied


def solve_remaining(visible_jobs, machine_free, now, objective="tardiness",
                    time_limit=0.1, seed=0):
    """Спланировать остаток ТОЛЬКО переданных заказов; будущее сюда не передаётся."""
    from ortools.sat.python import cp_model
    if objective not in ("tardiness", "makespan"):
        raise ValueError("Unknown objective")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive finite solve time required")
    if not visible_jobs:
        raise ValueError("No jobs to plan")
    if objective == "tardiness" and any(j.due_date is None for j in visible_jobs):
        raise ValueError("Tardiness objective requires deadlines")
    values = [now, *machine_free.values()]
    for j in visible_jobs:
        values.extend([j.release_time, j.ready_at])
        if j.due_date is not None:
            values.append(j.due_date)
        values.extend(o.duration for o in j.operations)
    if any(not math.isfinite(v) or int(v) != v for v in values):
        raise ValueError("CP-SAT requires integer ticks; quantize all methods first")

    model = cp_model.CpModel()
    remaining = [(j, o) for j in visible_jobs
                 for o in j.operations if o.start is None]
    # Гарантированно достаточный горизонт: дождаться доступности и выполнить всё
    # последовательно. Он не запрещает более короткое параллельное расписание.
    horizon = int(max([now, *machine_free.values(),
                       *(j.ready_at for j in visible_jobs)])
                  + sum(o.duration for _, o in remaining))
    starts, ends, intervals = {}, {}, {}
    completions = []
    for job in visible_jobs:
        previous_end = int(max(now, job.ready_at))
        unassigned = [o for o in job.operations if o.start is None]
        if not unassigned:
            completions.append((job, int(job.ready_at)))
            continue
        for op in unassigned:
            key = (job.job_id, op.op_index)
            # Начало и конец — неизвестные целые числа, которые найдёт решатель.
            earliest = int(max(now, job.release_time, machine_free[op.machine_id]))
            start = model.new_int_var(earliest, horizon, f"start_{key}")
            end = model.new_int_var(earliest, horizon, f"end_{key}")
            interval = model.new_interval_var(start, int(op.duration), end, f"op_{key}")
            model.add(start >= previous_end)  # технологическая последовательность
            previous_end = end
            starts[key], ends[key] = start, end
            intervals.setdefault(op.machine_id, []).append(interval)
        completions.append((job, previous_end))
    for own_intervals in intervals.values():
        model.add_no_overlap(own_intervals)  # станок делает только одну операцию

    if objective == "makespan":
        target = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(target, [end for _, end in completions])
    else:
        penalties = []
        for job, end in completions:
            upper = max(0, horizon - int(job.due_date))
            late = model.new_int_var(0, upper, f"late_{job.job_id}")
            model.add_max_equality(late, [0, end - int(job.due_date)])
            penalties.append(late)
        target = sum(penalties)
    model.minimize(target)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # UNKNOWN обычно означает, что времени не хватило даже на допустимое решение.
    return {
        "status": solver.status_name(status),
        "plan": {key: (solver.value(start), solver.value(ends[key]))
                 for key, start in starts.items()} if feasible else None,
        "objective": solver.objective_value if feasible else None,
        "lower_bound": solver.best_objective_bound if feasible else None,
    }


def _metrics(jobs):
    violations = validate_schedule(jobs)
    if violations:
        raise AssertionError(violations)
    work = sum(o.duration for j in jobs for o in j.operations)
    tardy = total_tardiness(jobs) if all(j.due_date is not None for j in jobs) else None
    return {
        "valid": True, "makespan": makespan(jobs), "total_tardiness": tardy,
        "normalized_tardiness": None if tardy is None else tardy / work,
        "schedule_sha256": schedule_fingerprint(jobs),
    }


def run_cp_offline(jobs, objective="tardiness", time_limit=1.0, seed=0):
    """Ориентир с полной информацией. В динамике он заранее знает будущие заказы."""
    copied = copy.deepcopy(jobs)
    mids = {o.machine_id for j in copied for o in j.operations}
    Environment(copied, [MachineState(mid) for mid in mids])
    solved = solve_remaining(copied, dict.fromkeys(mids, 0), 0,
                             objective, time_limit, seed)
    plan = solved.pop("plan")
    if plan is None:
        return {"valid": False, "method": "CP_OFFLINE", **solved,
                "note": "No feasible schedule found within limit"}
    for job in copied:
        for op in job.operations:
            op.start, op.end = plan[job.job_id, op.op_index]
    return {"method": "CP_OFFLINE", **_metrics(copied), **solved}


def run_cp_online(jobs, objective="tardiness", time_limit=0.05, seed=0,
                  capture_trace=False):
    """Перепланирование: выполненное не меняется, будущие заказы скрыты."""
    copied = copy.deepcopy(jobs)
    mids = sorted({o.machine_id for j in copied for o in j.operations})
    env = Environment(copied, [MachineState(mid) for mid in mids])
    statuses = Counter()
    fallbacks = calls = 0
    trace = []
    while not env.is_done():
        ready = env.ready_operations()
        if not ready:
            env.advance_time_to_next_event()
            continue
        visible = [j for j in copied if j.release_time <= env.time]
        solved = solve_remaining(
            visible, {m: s.free_at for m, s in env.machines.items()},
            env.time, objective, time_limit, seed)
        calls += 1
        statuses[solved["status"]] += 1
        plan = solved["plan"]
        started = []
        if plan is None:
            # Явный резервный путь: если решатель не успел, применить ATC.
            # Счётчик позволяет увидеть, сколько решений фактически принял ATC.
            fallbacks += 1
            for mid in mids:
                queue = [o for o in ready if o.machine_id == mid]
                if queue:
                    mean_p = sum(o.duration for o in queue) / len(queue)
                    op = min(queue, key=lambda o: (
                        -priority(env, o, "ATC_LOCAL", mean_p, 2.0),
                        o.job_id, o.op_index))
                    env.step(op)
                    started.append((op.job_id, op.op_index))
        else:
            for op in ready:
                if plan[op.job_id, op.op_index][0] == env.time:
                    env.step(op)
                    started.append((op.job_id, op.op_index))
        if capture_trace:
            trace.append({"time": env.time, "visible_job_ids": [j.job_id for j in visible],
                          "status": solved["status"], "started": started,
                          "fallback": plan is None})
        if not started:
            # Решатель может намеренно оставить станок свободным.
            # Новое поступление прерывает ожидание, но не было раскрыто решателю.
            future = [s.free_at for s in env.machines.values() if s.free_at > env.time]
            future += [j.release_time for j in copied if j.release_time > env.time]
            if plan:
                future += [start for start, _ in plan.values() if start > env.time]
            if not future:
                raise RuntimeError("No future event in CP plan")
            env.time = min(future)
    return {"method": "CP_ONLINE", **_metrics(copied),
            "solver_calls": calls, "solver_status_counts": dict(statuses),
            "fallback_calls": fallbacks, "trace": trace,
            "note": "Per-event optimality does not prove a globally optimal online schedule"}
