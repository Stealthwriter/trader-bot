from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

# Allow running tests from either project root or the tests directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import strategy as strategy_module
from strategy import SamaLiveStrategy, StrategyConfig


@dataclass
class FakePosition:
    ticket: int
    type: int
    volume: float
    symbol: str


@dataclass
class FakeTick:
    bid: float
    ask: float


@dataclass
class FakeOrderResult:
    retcode: int
    order: int
    comment: str


@dataclass
class FakeAccountInfo:
    equity: float


@dataclass
class FakeSymbolInfo:
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float


@dataclass
class FakeTerminalInfo:
    trade_allowed: bool
    tradeapi_disabled: bool


class FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._positions: list[FakePosition] = []
        self._next_ticket = 1000
        self.sent_orders: list[dict] = []
        self._equity = 10_000.0
        self._trade_allowed = True
        self._tradeapi_disabled = False

    def last_error(self):
        return (1, "Success")

    def account_info(self):
        return FakeAccountInfo(equity=self._equity)

    def symbol_info(self, symbol):
        if symbol != self.symbol:
            return None
        return FakeSymbolInfo(
            trade_contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )

    def terminal_info(self):
        return FakeTerminalInfo(
            trade_allowed=self._trade_allowed,
            tradeapi_disabled=self._tradeapi_disabled,
        )

    def positions_get(self, symbol=None):
        if symbol is not None and symbol != self.symbol:
            return []
        return list(self._positions)

    def symbol_info_tick(self, symbol):
        if symbol != self.symbol:
            return None
        return FakeTick(bid=100.0, ask=100.2)

    def order_send(self, request):
        self.sent_orders.append(dict(request))

        if request.get("symbol") != self.symbol:
            return FakeOrderResult(retcode=10013, order=0, comment="Invalid symbol")

        position_ticket = request.get("position")
        if position_ticket is not None:
            idx = next(
                (i for i, p in enumerate(self._positions) if int(p.ticket) == int(position_ticket)),
                None,
            )
            if idx is None:
                return FakeOrderResult(retcode=10013, order=0, comment="Position not found")
            closed_ticket = self._positions[idx].ticket
            del self._positions[idx]
            return FakeOrderResult(
                retcode=self.TRADE_RETCODE_DONE,
                order=int(closed_ticket),
                comment="Closed",
            )

        order_type = int(request["type"])
        volume = float(request["volume"])
        if self._positions:
            current = self._positions[0]
            if int(current.type) == order_type:
                current.volume += volume
                return FakeOrderResult(
                    retcode=self.TRADE_RETCODE_DONE,
                    order=int(current.ticket),
                    comment="Increased",
                )
            self._positions.clear()

        self._next_ticket += 1
        self._positions = [
            FakePosition(ticket=self._next_ticket, type=order_type, volume=volume, symbol=self.symbol)
        ]
        return FakeOrderResult(
            retcode=self.TRADE_RETCODE_DONE,
            order=int(self._next_ticket),
            comment="Opened",
        )


def make_candle(ts: int, price: float = 100.0) -> dict:
    return {
        "time": ts,
        "open": price,
        "high": price + 1.0,
        "low": price - 1.0,
        "close": price,
        "tick_volume": 1000.0,
        "spread": 0.0,
        "real_volume": 0.0,
    }


def make_initial_candles(count: int = 500, start_ts: int = 1_700_000_000) -> deque:
    out = deque(maxlen=500)
    for i in range(count):
        out.append(make_candle(start_ts + i * 3600, price=100.0 + i * 0.01))
    return out


@contextmanager
def patched_build_signal_frame(signal_by_marker: dict[float, dict]):
    original = strategy_module.build_signal_frame

    def fake_build_signal_frame(
        ohlcv: pd.DataFrame,
        ama_length: int,
        major_length: int,
        minor_length: int,
        slope_period: int,
        slope_in_range: float,
        flat_threshold: float,
    ) -> pd.DataFrame:
        del ama_length, major_length, minor_length, slope_period, slope_in_range, flat_threshold

        rows = len(ohlcv)
        idx = ohlcv.index
        frame = pd.DataFrame(
            {
                "sama": np.zeros(rows, dtype=float),
                "slope": np.zeros(rows, dtype=float),
                "bull": np.zeros(rows, dtype=bool),
                "bear": np.zeros(rows, dtype=bool),
                "chop": np.zeros(rows, dtype=bool),
                "long_flip": np.zeros(rows, dtype=bool),
                "short_flip": np.zeros(rows, dtype=bool),
            },
            index=idx,
        )

        last_idx = idx[-1]
        close_marker = round(float(ohlcv["Close"].iloc[-1]), 8)
        cfg = signal_by_marker.get(close_marker, {})

        for key, value in cfg.items():
            if key in frame.columns:
                frame.at[last_idx, key] = value
        return frame

    strategy_module.build_signal_frame = fake_build_signal_frame
    try:
        yield
    finally:
        strategy_module.build_signal_frame = original


class TestSamaStrategyWithFakeMT5(unittest.TestCase):
    def setUp(self) -> None:
        self.symbol = "ETHUSDr"
        self.mt5 = FakeMT5(self.symbol)
        self.base_cfg = StrategyConfig(
            position_size=0.1,
            warmup_bars=0,
            chop_debounce=True,
            chop_debounce_bars=1,
            use_hard_stop_loss=False,
        )

    def test_startup_adopt_single_position_and_abort_on_multiple(self):
        candles = make_initial_candles()

        self.mt5._positions = [
            FakePosition(
                ticket=1,
                type=self.mt5.POSITION_TYPE_BUY,
                volume=0.1,
                symbol=self.symbol,
            )
        ]
        strategy = SamaLiveStrategy(self.mt5, self.symbol, self.base_cfg)
        with patched_build_signal_frame({}):
            strategy.initialize(candles)
        self.assertEqual(strategy.position_side, "long")

        self.mt5._positions = [
            FakePosition(ticket=1, type=self.mt5.POSITION_TYPE_BUY, volume=0.1, symbol=self.symbol),
            FakePosition(ticket=2, type=self.mt5.POSITION_TYPE_SELL, volume=0.1, symbol=self.symbol),
        ]
        strategy2 = SamaLiveStrategy(self.mt5, self.symbol, self.base_cfg)
        with patched_build_signal_frame({}):
            with self.assertRaises(RuntimeError):
                strategy2.initialize(candles)

    def test_entry_reverse_and_chop_priority(self):
        candles = make_initial_candles()
        strategy = SamaLiveStrategy(self.mt5, self.symbol, self.base_cfg)

        t1 = int(candles[-1]["time"]) + 3600
        t2 = t1 + 3600
        t3 = t2 + 3600
        t4 = t3 + 3600
        t5 = t4 + 3600

        p1 = 1000.1
        p2 = 1000.2
        p3 = 1000.3
        p4 = 1000.4
        p5 = 1000.5

        signal_map = {
            p1: {"long_flip": True, "short_flip": False, "chop": False},
            p2: {"long_flip": False, "short_flip": True, "chop": False},
            p3: {
                "long_flip": True,
                "short_flip": False,
                "chop": True,
            },  # chop exit should win, no same-bar reverse
            p4: {"long_flip": False, "short_flip": False, "chop": False},
            p5: {"long_flip": True, "short_flip": True, "chop": False},  # XOR false => no trade
        }

        with patched_build_signal_frame(signal_map):
            strategy.initialize(candles)

            candles.append(make_candle(t1, p1))
            strategy.on_new_closed_bar(candles)
            self.assertEqual(strategy.position_side, "long")
            self.assertEqual(len(self.mt5.sent_orders), 1)

            candles.append(make_candle(t2, p2))
            strategy.on_new_closed_bar(candles)
            self.assertEqual(strategy.position_side, "short")
            self.assertEqual(len(self.mt5.sent_orders), 3)  # close + new entry

            candles.append(make_candle(t3, p3))
            strategy.on_new_closed_bar(candles)
            self.assertEqual(strategy.position_side, "flat")
            self.assertEqual(len(self.mt5.sent_orders), 4)  # only chop close

            candles.append(make_candle(t4, p4))
            strategy.on_new_closed_bar(candles)
            self.assertEqual(strategy.position_side, "flat")
            self.assertEqual(len(self.mt5.sent_orders), 4)

            candles.append(make_candle(t5, p5))
            strategy.on_new_closed_bar(candles)
            self.assertEqual(strategy.position_side, "flat")
            self.assertEqual(len(self.mt5.sent_orders), 4)

        self.assertEqual(len(strategy.trades), 4)
        self.assertEqual([t["reason"] for t in strategy.trades], ["entry_long", "reverse_to_short", "entry_short", "chop_exit"])
        self.assertEqual(strategy.trades[0]["side"], "long")
        self.assertEqual(strategy.trades[-1]["side"], "long")

    def test_warmup_guard_blocks_trades(self):
        candles = make_initial_candles()
        cfg = StrategyConfig(
            position_size=0.1,
            warmup_bars=9999,
            chop_debounce=True,
            chop_debounce_bars=1,
        )
        strategy = SamaLiveStrategy(self.mt5, self.symbol, cfg)

        t1 = int(candles[-1]["time"]) + 3600
        p1 = 2000.1
        signal_map = {
            p1: {"long_flip": True, "short_flip": False, "chop": False},
        }

        with patched_build_signal_frame(signal_map):
            strategy.initialize(candles)
            candles.append(make_candle(t1, p1))
            strategy.on_new_closed_bar(candles)

        self.assertEqual(strategy.position_side, "flat")
        self.assertEqual(len(self.mt5.sent_orders), 0)

    def test_autotrading_disabled_blocks_order_submission(self):
        candles = make_initial_candles()
        strategy = SamaLiveStrategy(self.mt5, self.symbol, self.base_cfg)

        t1 = int(candles[-1]["time"]) + 3600
        p1 = 3000.1
        signal_map = {
            p1: {"long_flip": True, "short_flip": False, "chop": False},
        }

        self.mt5._trade_allowed = False
        with patched_build_signal_frame(signal_map):
            strategy.initialize(candles)
            candles.append(make_candle(t1, p1))
            strategy.on_new_closed_bar(candles)

        self.assertEqual(strategy.position_side, "flat")
        self.assertEqual(len(self.mt5.sent_orders), 0)
        self.assertEqual(len(strategy.trades), 0)


if __name__ == "__main__":
    unittest.main()
