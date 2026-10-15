"""Правила торгов для одного свободного станка.

Этот модуль не знает о расписании. Он получает ставки и отвечает:
кто победил, сколько платит и сколько раундов потребовалось.
Кредиты условные; никакие реальные деньги не используются.
"""
import math
from dataclasses import dataclass

PROTOCOLS = ("no_payment", "first_price", "second_price", "ascending")


@dataclass(frozen=True)
class Bid:
    # frozen запрещает изменять сообщение после его создания.
    job_id: int
    amount: float


@dataclass(frozen=True)
class Award:
    winner_id: int
    payment: float
    rounds: int
    messages: int
    reason: str


def clear_auction(bids, protocol, price_step=0.1, max_rounds=1000):
    """Провести один аукцион; равные ставки разрешаются по номеру заказа."""
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown auction protocol")
    if not bids or len({b.job_id for b in bids}) != len(bids):
        raise ValueError("Nonempty bids with unique job IDs required")
    if any(not math.isfinite(b.amount) or b.amount < 0 for b in bids):
        raise ValueError("Bids must be finite and nonnegative")
    if not math.isfinite(price_step) or price_step <= 0:
        raise ValueError("Price step must be positive")
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("Positive integer round limit required")

    # Минус превращает сортировку по возрастанию в сортировку по убыванию ставки.
    ordered = sorted(bids, key=lambda b: (-b.amount, b.job_id))
    # Без конкуренции никого не нужно вытеснять: операция назначается бесплатно.
    if len(ordered) == 1:
        return Award(ordered[0].job_id, 0.0, 0, 2, "single_candidate")
    if protocol != "ascending":
        if protocol == "first_price":
            payment = ordered[0].amount
        elif protocol == "second_price":
            payment = ordered[1].amount
        else:
            payment = 0.0  # контроль: тот же выбор, но бюджет не расходуется
        # Ставка от каждого + уведомление результата каждому.
        return Award(ordered[0].job_id, payment, 1, 2 * len(bids), protocol)

    # Повышающийся ценовой аукцион. amount здесь — личный предел участия.
    # Участники отвечают на цены; точный предел считается моделью их поведения.
    # Это симуляция ценовых часов, а не отдельные сетевые процессы.
    active = list(ordered)
    price = 0.0
    messages = 0
    for round_number in range(1, max_rounds + 1):
        next_price = round_number * price_step
        # Объявление новой цены и ответ каждого ещё участвующего заказа.
        messages += 2 * len(active)
        willing = [b for b in active if b.amount >= next_price]
        if len(willing) == 1:
            return Award(willing[0].job_id, next_price, round_number,
                         messages + len(bids), "one_remaining")
        if not willing:
            # Шаг мог перескочить пределы всех оставшихся одновременно.
            # Победитель определяется только по ID, а НЕ по скрытым пределам.
            # Он платит предыдущую принятую цену. Так шаг цены влияет на результат.
            winner = min(active, key=lambda b: b.job_id)
            return Award(winner.job_id, price, round_number,
                         messages + len(bids), "simultaneous_exit_id_tie")
        active, price = willing, next_price
    # Чтобы торги не длились бесконечно, есть заранее объявленный предел.
    # При его достижении также применяется прозрачное правило по ID.
    winner = min(active, key=lambda b: b.job_id)
    return Award(winner.job_id, price, max_rounds,
                 messages + len(bids), "round_cap_id_tie")
