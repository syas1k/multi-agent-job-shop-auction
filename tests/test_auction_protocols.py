"""Проверяем смысл торгов, а не только отсутствие ошибок исполнения."""
import copy

import pytest

from mas_jssp.auction.protocols import Bid, PROTOCOLS, clear_auction
from mas_jssp.agents.budget_auction import run_budget_auction
from mas_jssp.environment.environment import Job, Operation
from mas_jssp.learning_data import generate_scenario
from mas_jssp.utils.loaders import load_jssp_instance


def test_first_and_second_price_same_winner_different_payment():
    bids = [Bid(1, 3), Bid(2, 5), Bid(3, 2)]
    first = clear_auction(bids, "first_price")
    second = clear_auction(bids, "second_price")
    assert first.winner_id == second.winner_id == 2
    assert first.payment == 5
    assert second.payment == 3


def test_clock_auction_rounds_and_payment():
    result = clear_auction([Bid(1, 3), Bid(2, 5)], "ascending", price_step=1)
    assert result.winner_id == 2
    assert result.payment == 4
    assert result.rounds == 4


def test_clock_simultaneous_exit_does_not_read_private_values_for_tie():
    # Оба отказались от первой цены: побеждает меньший ID, а не больший предел.
    result = clear_auction([Bid(1, 0.2), Bid(2, 0.3)], "ascending", price_step=1)
    assert result.winner_id == 1 and result.payment == 0
    assert result.reason == "simultaneous_exit_id_tie"


def test_clock_round_limit_never_charges_more_than_accepted():
    result = clear_auction([Bid(1, 10), Bid(2, 20)], "ascending",
                           price_step=1, max_rounds=2)
    assert result.winner_id == 1 and result.payment == 2
    assert result.reason == "round_cap_id_tie"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_single_and_zero_bids(protocol):
    assert clear_auction([Bid(3, 5)], protocol).payment == 0
    zero = clear_auction([Bid(3, 0), Bid(1, 0)], protocol)
    assert zero.winner_id == 1 and zero.payment == 0


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("credits", [0, 1])
def test_budget_conservation_completion_and_unchanged_input(protocol, credits):
    jobs = generate_scenario(load_jssp_instance("data/benchmarks/ft06.txt"), "ft06", 420000)
    before = copy.deepcopy(jobs)
    result = run_budget_auction(jobs, protocol, credits, capture_trace=True)
    assert jobs == before and result["valid"]
    assert result["completed_jobs"] == len(jobs)
    initial = sum(len(j.operations) for j in jobs) * credits
    assert result["credits_spent"] + sum(result["remaining_budgets"].values()) == pytest.approx(initial)
    assert all(b >= 0 for b in result["remaining_budgets"].values())
    for event in result["trace"]:
        assert event["award"]["payment"] <= event["budget_before"] + 1e-9
    assert run_budget_auction(jobs, protocol, credits)["schedule_sha256"] == result["schedule_sha256"]


def test_payment_changes_future_schedule():
    # Три одинаковых заказа снова встречаются на том же станке.
    # После первой победы остаток бюджета отличается между протоколами.
    jobs = [Job(j, [Operation(j, 0, 0, 1), Operation(j, 1, 0, 1)],
                due_date=d) for j, d in enumerate([2, 4, 6])]
    first = run_budget_auction(jobs, "first_price", capture_trace=True)
    second = run_budget_auction(jobs, "second_price", capture_trace=True)
    assert first["trace"][0]["award"]["winner_id"] == second["trace"][0]["award"]["winner_id"]
    assert first["trace"][0]["award"]["payment"] != second["trace"][0]["award"]["payment"]
    # Полный пример допускает совпадение порядка: расход и будущая ставка точно различаются.
    assert first["trace"][1]["bids_or_private_limits"] != second["trace"][1]["bids_or_private_limits"]


@pytest.mark.parametrize("bids", [[], [Bid(1, -1)], [Bid(1, float("nan"))],
                                [Bid(1, 2), Bid(1, 3)]])
def test_invalid_bids(bids):
    with pytest.raises(ValueError):
        clear_auction(bids, "first_price")
