"""
Price-Time Priority (FIFO) Limit Order Book Engine with 1-Tick Execution Latency.

Implements L3 order queue tracking using collections.deque at discrete price ticks,
ensuring exact FIFO lot conservation and zero lookahead bias by buffering newly
submitted strategy orders across a configurable tick latency horizon.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal, Optional, Tuple

Side = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET"]


@dataclass
class Order:
    order_id: int
    side: Side
    price: float
    quantity: float
    timestamp: int
    order_type: OrderType = "LIMIT"
    is_strategy: bool = False
    remaining_qty: float = field(init=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be strictly positive.")
        self.remaining_qty = float(self.quantity)


@dataclass(frozen=True)
class TradeFill:
    timestamp: int
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: float
    aggressor_side: Side
    strategy_involved: bool


class LimitOrderBook:
    """
    Event-driven Price-Time Priority (FIFO) Limit Order Book.

    Attributes
    ----------
    tick_size : float
        Minimum price increment used to quantize price levels.
    latency_ticks : int
        Number of discrete tick updates an incoming strategy order must wait
        in the flight buffer before hitting the matching engine (default=1).
    """

    def __init__(self, tick_size: float = 0.01, latency_ticks: int = 1) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive.")
        if latency_ticks < 0:
            raise ValueError("latency_ticks must be non-negative.")

        self.tick_size: float = tick_size
        self.latency_ticks: int = latency_ticks
        self.current_tick: int = 0

        # Price tick integer index -> FIFO deque of active Orders
        self._bids: Dict[int, Deque[Order]] = {}
        self._asks: Dict[int, Deque[Order]] = {}
        self._order_map: Dict[int, Tuple[Side, int, Order]] = {}

        # Latency queue: list of (release_tick, Order)
        self._in_flight_buffer: Deque[Tuple[int, Order]] = deque()
        self.fills: List[TradeFill] = []

    def _to_tick(self, price: float) -> int:
        return int(round(price / self.tick_size))

    def _to_price(self, tick_idx: int) -> float:
        return round(tick_idx * self.tick_size, 6)

    def best_bid_tick(self) -> Optional[int]:
        active = [k for k, q in self._bids.items() if len(q) > 0]
        return max(active) if active else None

    def best_ask_tick(self) -> Optional[int]:
        active = [k for k, q in self._asks.items() if len(q) > 0]
        return min(active) if active else None

    def best_bid(self) -> Optional[float]:
        t = self.best_bid_tick()
        return self._to_price(t) if t is not None else None

    def best_ask(self) -> Optional[float]:
        t = self.best_ask_tick()
        return self._to_price(t) if t is not None else None

    def mid_price(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            return 0.5 * (bb + ba)
        return None

    def spread(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            return max(0.0, ba - bb)
        return None

    def queue_imbalance(self, depth_levels: int = 3) -> float:
        """
        Computes top-N level volume imbalance I_t in [-1.0, +1.0].
        """
        bid_ticks = sorted(
            [k for k, q in self._bids.items() if len(q) > 0], reverse=True
        )[:depth_levels]
        ask_ticks = sorted(
            [k for k, q in self._asks.items() if len(q) > 0]
        )[:depth_levels]

        v_bid = sum(sum(o.remaining_qty for o in self._bids[k]) for k in bid_ticks)
        v_ask = sum(sum(o.remaining_qty for o in self._asks[k]) for k in ask_ticks)
        denom = v_bid + v_ask
        if denom <= 1e-12:
            return 0.0
        return float((v_bid - v_ask) / denom)

    def submit_order(self, order: Order) -> List[TradeFill]:
        """
        Submits an order. Strategy orders respect `latency_ticks` buffering;
        background market feed orders execute at the current tick.
        """
        if order.is_strategy and self.latency_ticks > 0:
            release_tick = self.current_tick + self.latency_ticks
            self._in_flight_buffer.append((release_tick, order))
            return []
        return self._match_or_enqueue(order)

    def advance_tick(self, new_tick: Optional[int] = None) -> List[TradeFill]:
        """
        Advances the book clock and releases any strategy orders whose
        latency horizon has elapsed.
        """
        if new_tick is not None:
            if new_tick < self.current_tick:
                raise ValueError(
                    "Monotonic clock violation: lookahead or backward tick detected."
                )
            self.current_tick = new_tick
        else:
            self.current_tick += 1

        released_fills: List[TradeFill] = []
        while (
            self._in_flight_buffer
            and self._in_flight_buffer[0][0] <= self.current_tick
        ):
            _, buffered_order = self._in_flight_buffer.popleft()
            buffered_order.timestamp = self.current_tick
            released_fills.extend(self._match_or_enqueue(buffered_order))
        return released_fills

    def cancel_order(self, order_id: int) -> bool:
        if order_id not in self._order_map:
            return False
        side, tick_idx, target_order = self._order_map.pop(order_id)
        book_side = self._bids if side == "BUY" else self._asks
        queue = book_side.get(tick_idx)
        if queue and target_order in queue:
            queue.remove(target_order)
            if len(queue) == 0:
                del book_side[tick_idx]
            return True
        return False

    def _match_or_enqueue(self, incoming: Order) -> List[TradeFill]:
        new_fills: List[TradeFill] = []
        incoming_tick = (
            self._to_tick(incoming.price)
            if incoming.order_type == "LIMIT"
            else None
        )

        if incoming.side == "BUY":
            while incoming.remaining_qty > 1e-9:
                best_ask_t = self.best_ask_tick()
                if best_ask_t is None:
                    break
                if (
                    incoming.order_type == "LIMIT"
                    and incoming_tick is not None
                    and incoming_tick < best_ask_t
                ):
                    break

                ask_queue = self._asks[best_ask_t]
                resting = ask_queue[0]
                fill_qty = min(incoming.remaining_qty, resting.remaining_qty)
                exec_price = self._to_price(best_ask_t)

                incoming.remaining_qty -= fill_qty
                resting.remaining_qty -= fill_qty

                fill = TradeFill(
                    timestamp=self.current_tick,
                    buy_order_id=incoming.order_id,
                    sell_order_id=resting.order_id,
                    price=exec_price,
                    quantity=fill_qty,
                    aggressor_side="BUY",
                    strategy_involved=(incoming.is_strategy or resting.is_strategy),
                )
                self.fills.append(fill)
                new_fills.append(fill)

                if resting.remaining_qty <= 1e-9:
                    ask_queue.popleft()
                    self._order_map.pop(resting.order_id, None)
                    if len(ask_queue) == 0:
                        del self._asks[best_ask_t]

            if (
                incoming.order_type == "LIMIT"
                and incoming.remaining_qty > 1e-9
                and incoming_tick is not None
            ):
                if incoming_tick not in self._bids:
                    self._bids[incoming_tick] = deque()
                self._bids[incoming_tick].append(incoming)
                self._order_map[incoming.order_id] = ("BUY", incoming_tick, incoming)

        else:  # incoming.side == "SELL"
            while incoming.remaining_qty > 1e-9:
                best_bid_t = self.best_bid_tick()
                if best_bid_t is None:
                    break
                if (
                    incoming.order_type == "LIMIT"
                    and incoming_tick is not None
                    and incoming_tick > best_bid_t
                ):
                    break

                bid_queue = self._bids[best_bid_t]
                resting = bid_queue[0]
                fill_qty = min(incoming.remaining_qty, resting.remaining_qty)
                exec_price = self._to_price(best_bid_t)

                incoming.remaining_qty -= fill_qty
                resting.remaining_qty -= fill_qty

                fill = TradeFill(
                    timestamp=self.current_tick,
                    buy_order_id=resting.order_id,
                    sell_order_id=incoming.order_id,
                    price=exec_price,
                    quantity=fill_qty,
                    aggressor_side="SELL",
                    strategy_involved=(incoming.is_strategy or resting.is_strategy),
                )
                self.fills.append(fill)
                new_fills.append(fill)

                if resting.remaining_qty <= 1e-9:
                    bid_queue.popleft()
                    self._order_map.pop(resting.order_id, None)
                    if len(bid_queue) == 0:
                        del self._bids[best_bid_t]

            if (
                incoming.order_type == "LIMIT"
                and incoming.remaining_qty > 1e-9
                and incoming_tick is not None
            ):
                if incoming_tick not in self._asks:
                    self._asks[incoming_tick] = deque()
                self._asks[incoming_tick].append(incoming)
                self._order_map[incoming.order_id] = ("SELL", incoming_tick, incoming)

        return new_fills

