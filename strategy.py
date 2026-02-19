from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd


PositionSide = Literal["flat", "long", "short"]


def compute_sama(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ama_length: int,
    minor_length: int,
    major_length: int,
) -> pd.Series:
    min_alpha = 2.0 / (float(minor_length) + 1.0)
    maj_alpha = 2.0 / (float(major_length) + 1.0)

    lookback = max(int(ama_length) + 1, 1)
    hh = high.rolling(window=lookback, min_periods=lookback).max()
    ll = low.rolling(window=lookback, min_periods=lookback).min()

    denom = hh - ll
    mult = pd.Series(np.nan, index=close.index, dtype=float)
    valid = denom != 0.0
    mult.loc[valid] = (2.0 * close.loc[valid] - ll.loc[valid] - hh.loc[valid]).abs() / denom.loc[valid]
    mult.loc[~valid] = 0.0

    alpha = mult * (min_alpha - maj_alpha) + maj_alpha
    final_alpha = alpha.pow(2)

    out = np.full(len(close), np.nan, dtype=float)
    prev_ama = np.nan
    close_values = close.to_numpy(dtype=float, copy=False)
    alpha_values = final_alpha.to_numpy(dtype=float, copy=False)

    for i in range(len(close_values)):
        price = close_values[i]
        a = alpha_values[i]
        if not np.isfinite(price) or not np.isfinite(a):
            continue
        if not np.isfinite(prev_ama):
            current = price
        else:
            current = (price - prev_ama) * a + prev_ama
        out[i] = current
        prev_ama = current

    return pd.Series(out, index=close.index, dtype=float)


def compute_slope_angle(
    ma: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    slope_period: int,
    slope_in_range: float,
) -> pd.Series:
    period = max(int(slope_period), 1)
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    spread = highest_high - lowest_low

    slope_range = pd.Series(np.nan, index=close.index, dtype=float)
    valid_spread = spread != 0.0
    slope_range.loc[valid_spread] = (
        float(slope_in_range) / spread.loc[valid_spread] * lowest_low.loc[valid_spread]
    )

    close_nonzero = close.replace(0.0, np.nan)
    dt = (ma.shift(2) - ma) / close_nonzero * slope_range
    c = np.sqrt(1.0 + dt.pow(2))
    inv_c = (1.0 / c).clip(lower=-1.0, upper=1.0)
    x_angle = np.round(np.degrees(np.arccos(inv_c)))
    ma_angle = np.where(dt > 0.0, -x_angle, x_angle)
    return pd.Series(ma_angle, index=close.index, dtype=float)


def build_signal_frame(
    ohlcv: pd.DataFrame,
    ama_length: int,
    major_length: int,
    minor_length: int,
    slope_period: int,
    slope_in_range: float,
    flat_threshold: float,
) -> pd.DataFrame:
    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]

    sama = compute_sama(
        close=close,
        high=high,
        low=low,
        ama_length=ama_length,
        minor_length=minor_length,
        major_length=major_length,
    )
    slope = compute_slope_angle(
        ma=sama,
        close=close,
        high=high,
        low=low,
        slope_period=slope_period,
        slope_in_range=slope_in_range,
    )

    flat = float(flat_threshold)
    bull = slope > flat
    bear = slope <= -flat
    chop = ~bull & ~bear
    long_flip = bull & bull.shift(1).ne(True)
    short_flip = bear & bear.shift(1).ne(True)

    return pd.DataFrame(
        {
            "sama": sama,
            "slope": slope,
            "bull": bull.astype(bool),
            "bear": bear.astype(bool),
            "chop": chop.astype(bool),
            "long_flip": long_flip.astype(bool),
            "short_flip": short_flip.astype(bool),
        },
        index=ohlcv.index,
    )


@dataclass
class StrategyConfig:
    position_size: float = 0.0025
    ama_length: int = 191
    major_length: int = 28
    minor_length: int = 6
    slope_period: int = 109
    slope_in_range: float = 30.0
    flat_threshold: float = 30.0
    warmup_bars: int = 400
    chop_debounce: bool = True
    chop_debounce_bars: int = 3
    use_hard_stop_loss: bool = False
    hard_stop_loss_pct: float = 0.04
    order_deviation: int = 20
    magic: int = 1802407

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        return cls(
            position_size=float(data.get("position_size", cls.position_size)),
            ama_length=int(data.get("ama_length", cls.ama_length)),
            major_length=int(data.get("major_length", cls.major_length)),
            minor_length=int(data.get("minor_length", cls.minor_length)),
            slope_period=int(data.get("slope_period", cls.slope_period)),
            slope_in_range=float(data.get("slope_in_range", cls.slope_in_range)),
            flat_threshold=float(data.get("flat_threshold", cls.flat_threshold)),
            warmup_bars=max(int(data.get("warmup_bars", cls.warmup_bars)), 0),
            chop_debounce=bool(data.get("chop_debounce", cls.chop_debounce)),
            chop_debounce_bars=max(int(data.get("chop_debounce_bars", cls.chop_debounce_bars)), 1),
            use_hard_stop_loss=bool(data.get("use_hard_stop_loss", cls.use_hard_stop_loss)),
            hard_stop_loss_pct=float(data.get("hard_stop_loss_pct", cls.hard_stop_loss_pct)),
            order_deviation=int(data.get("order_deviation", cls.order_deviation)),
            magic=int(data.get("magic", cls.magic)),
        )


class SamaLiveStrategy:
    def __init__(self, mt5: Any, symbol: str, config: StrategyConfig) -> None:
        self.mt5 = mt5
        self.symbol = symbol
        self.cfg = config

        self.chop_streak = 0
        self.last_processed_bar_time: int | None = None
        self.position_side: PositionSide = "flat"

        self._events: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []

        self._order_type_buy = int(getattr(self.mt5, "ORDER_TYPE_BUY", 0))
        self._order_type_sell = int(getattr(self.mt5, "ORDER_TYPE_SELL", 1))
        self._position_type_buy = int(getattr(self.mt5, "POSITION_TYPE_BUY", self._order_type_buy))
        self._position_type_sell = int(getattr(self.mt5, "POSITION_TYPE_SELL", self._order_type_sell))
        self._trade_action_deal = int(getattr(self.mt5, "TRADE_ACTION_DEAL", 1))
        self._order_time_gtc = int(getattr(self.mt5, "ORDER_TIME_GTC", 0))
        self._order_filling_ioc = int(getattr(self.mt5, "ORDER_FILLING_IOC", 1))
        self._retcode_done = int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009))
        self._retcode_placed = int(getattr(self.mt5, "TRADE_RETCODE_PLACED", 10008))
        self._retcode_client_disable = int(
            getattr(self.mt5, "TRADE_RETCODE_CLIENT_DISABLES_AT", 10027)
        )

    @classmethod
    def from_config(cls, mt5: Any, symbol: str, config_data: dict[str, Any]) -> "SamaLiveStrategy":
        return cls(mt5=mt5, symbol=symbol, config=StrategyConfig.from_dict(config_data))

    def initialize(self, candles: Iterable[Any]) -> None:
        if not (0.0 < float(self.cfg.position_size) < 1.0):
            raise ValueError("strategy.position_size must satisfy 0 < position_size < 1.")
        if self.cfg.use_hard_stop_loss and not (0.0 < self.cfg.hard_stop_loss_pct < 1.0):
            raise ValueError(
                "strategy.hard_stop_loss_pct must satisfy 0 < hard_stop_loss_pct < 1 when enabled."
            )

        ohlcv = self._candles_to_ohlcv(candles)
        if ohlcv.empty:
            raise ValueError("No candle data provided to strategy initialization.")

        self._bootstrap_chop_streak(ohlcv)
        self.last_processed_bar_time = int(ohlcv["time"].iloc[-1])
        self.on_startup_adopt_position()

        bar_date, bar_clock = self._bar_date_time(self.last_processed_bar_time)
        self._log(
            "INFO",
            "startup.initialized",
            symbol=self.symbol,
            side=self.position_side,
            chop_streak=self.chop_streak,
            date=bar_date,
            time=bar_clock,
        )

    def on_startup_adopt_position(self) -> None:
        positions = self.mt5.positions_get(symbol=self.symbol)
        if positions is None:
            raise RuntimeError(
                f"positions_get({self.symbol}) failed on startup, last_error={self.mt5.last_error()}"
            )

        if len(positions) == 0:
            self.position_side = "flat"
            self._log("INFO", "startup.adopt", symbol=self.symbol, adopted_side="flat")
            return

        if len(positions) > 1:
            raise RuntimeError(
                f"Found {len(positions)} open positions for {self.symbol}. "
                "Aborting as requested (single-position mode)."
            )

        self.position_side = self._position_type_to_side(int(positions[0].type))
        self._log(
            "INFO",
            "startup.adopt",
            symbol=self.symbol,
            adopted_side=self.position_side,
            ticket=getattr(positions[0], "ticket", "n/a"),
            volume=float(positions[0].volume),
        )

    def on_new_closed_bar(self, candles: Iterable[Any]) -> None:
        ohlcv = self._candles_to_ohlcv(candles)
        if ohlcv.empty:
            return

        bar_time = int(ohlcv["time"].iloc[-1])
        if self.last_processed_bar_time is not None and bar_time <= self.last_processed_bar_time:
            return

        frame = build_signal_frame(
            ohlcv=ohlcv[["Open", "High", "Low", "Close", "Volume"]],
            ama_length=self.cfg.ama_length,
            major_length=self.cfg.major_length,
            minor_length=self.cfg.minor_length,
            slope_period=self.cfg.slope_period,
            slope_in_range=self.cfg.slope_in_range,
            flat_threshold=self.cfg.flat_threshold,
        )
        last = frame.iloc[-1]

        bars_seen = len(ohlcv)
        if bars_seen <= self.cfg.warmup_bars:
            self.last_processed_bar_time = bar_time
            return

        self._sync_position_side()
        side_before = self.position_side

        chop_now = bool(last["chop"])
        if chop_now:
            self.chop_streak += 1
        else:
            self.chop_streak = 0

        long_signal = bool(last["long_flip"])
        short_signal = bool(last["short_flip"])
        bar_date, bar_clock = self._bar_date_time(bar_time)
        self._log(
            "INFO",
            "signal.eval",
            date=bar_date,
            time=bar_clock,
            side_before=side_before,
            slope=float(last["slope"]),
            long_flip=int(long_signal),
            short_flip=int(short_signal),
            chop=int(chop_now),
            chop_streak=self.chop_streak,
        )

        if self.position_side != "flat":
            if self.cfg.chop_debounce:
                if self.chop_streak >= self.cfg.chop_debounce_bars:
                    self._record_event("chop_exit", bar_time, {"debounce_streak": self.chop_streak})
                    self._close_single_position(reason="chop_exit", event_time=bar_time)
                    self.last_processed_bar_time = bar_time
                    return
            elif chop_now:
                self._record_event("chop_exit", bar_time, {"debounce_streak": self.chop_streak})
                self._close_single_position(reason="chop_exit", event_time=bar_time)
                self.last_processed_bar_time = bar_time
                return

        if long_signal == short_signal:
            self.last_processed_bar_time = bar_time
            return

        if not (0.0 < float(self.cfg.position_size) < 1.0):
            self.last_processed_bar_time = bar_time
            self._log(
                "WARN",
                "signal.eval",
                error_code="invalid_position_size",
                position_size=float(self.cfg.position_size),
            )
            return

        close_price = float(ohlcv["Close"].iloc[-1])
        if long_signal and self.position_side in ("flat", "short"):
            self._record_event("long_flip", bar_time, {"from_side": self.position_side})
            self._open_or_reverse(target_side="long", signal_close=close_price, event_time=bar_time)
        elif short_signal and self.position_side in ("flat", "long"):
            self._record_event("short_flip", bar_time, {"from_side": self.position_side})
            self._open_or_reverse(target_side="short", signal_close=close_price, event_time=bar_time)

        self.last_processed_bar_time = bar_time

    def replay_events(self, candles: Iterable[Any], initial_side: PositionSide = "flat") -> list[dict[str, Any]]:
        ohlcv = self._candles_to_ohlcv(candles)
        if ohlcv.empty:
            return []

        frame = build_signal_frame(
            ohlcv=ohlcv[["Open", "High", "Low", "Close", "Volume"]],
            ama_length=self.cfg.ama_length,
            major_length=self.cfg.major_length,
            minor_length=self.cfg.minor_length,
            slope_period=self.cfg.slope_period,
            slope_in_range=self.cfg.slope_in_range,
            flat_threshold=self.cfg.flat_threshold,
        )

        events: list[dict[str, Any]] = []
        side: PositionSide = initial_side
        chop_streak = 0

        for i in range(len(ohlcv)):
            bars_seen = i + 1
            bar_time = int(ohlcv["time"].iloc[i])
            if bars_seen <= self.cfg.warmup_bars:
                continue

            chop_now = bool(frame["chop"].iloc[i])
            if chop_now:
                chop_streak += 1
            else:
                chop_streak = 0

            if side != "flat":
                if self.cfg.chop_debounce:
                    if chop_streak >= self.cfg.chop_debounce_bars:
                        events.append(
                            {"time": bar_time, "event": "chop_exit", "from": side, "to": "flat"}
                        )
                        side = "flat"
                        continue
                elif chop_now:
                    events.append({"time": bar_time, "event": "chop_exit", "from": side, "to": "flat"})
                    side = "flat"
                    continue

            long_signal = bool(frame["long_flip"].iloc[i])
            short_signal = bool(frame["short_flip"].iloc[i])
            if long_signal == short_signal:
                continue

            if not (0.0 < float(self.cfg.position_size) < 1.0):
                continue

            if long_signal and side in ("flat", "short"):
                old_side = side
                side = "long"
                events.append({"time": bar_time, "event": "long_flip", "from": old_side, "to": side})
            elif short_signal and side in ("flat", "long"):
                old_side = side
                side = "short"
                events.append({"time": bar_time, "event": "short_flip", "from": old_side, "to": side})

        return events

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    @property
    def trades(self) -> list[dict[str, Any]]:
        return list(self._trades)

    def _bootstrap_chop_streak(self, ohlcv: pd.DataFrame) -> None:
        frame = build_signal_frame(
            ohlcv=ohlcv[["Open", "High", "Low", "Close", "Volume"]],
            ama_length=self.cfg.ama_length,
            major_length=self.cfg.major_length,
            minor_length=self.cfg.minor_length,
            slope_period=self.cfg.slope_period,
            slope_in_range=self.cfg.slope_in_range,
            flat_threshold=self.cfg.flat_threshold,
        )

        streak = 0
        for i in range(len(frame)):
            bars_seen = i + 1
            if bars_seen <= self.cfg.warmup_bars:
                continue
            if bool(frame["chop"].iloc[i]):
                streak += 1
            else:
                streak = 0
        self.chop_streak = streak

    def _candles_to_ohlcv(self, candles: Iterable[Any]) -> pd.DataFrame:
        rows = list(candles)
        if len(rows) == 0:
            return pd.DataFrame()

        parsed_rows: list[dict[str, Any]] = [self._bar_to_dict(row) for row in rows]
        normalized_rows: list[dict[str, Any]] = []
        for row in parsed_rows:
            lower_map = {str(k).lower(): v for k, v in row.items()}
            normalized_rows.append(
                {
                    "time": lower_map.get("time"),
                    "open": lower_map.get("open"),
                    "high": lower_map.get("high"),
                    "low": lower_map.get("low"),
                    "close": lower_map.get("close"),
                    "tick_volume": lower_map.get("tick_volume", lower_map.get("volume", 0.0)),
                }
            )

        out = pd.DataFrame(normalized_rows)
        expected = {"time", "open", "high", "low", "close"}
        missing = sorted(expected - set(out.columns))
        if missing:
            raise ValueError(f"Candle data missing required field(s): {missing}")

        out = pd.DataFrame(
            {
                "time": pd.to_numeric(out["time"], errors="coerce"),
                "Open": pd.to_numeric(out["open"], errors="coerce"),
                "High": pd.to_numeric(out["high"], errors="coerce"),
                "Low": pd.to_numeric(out["low"], errors="coerce"),
                "Close": pd.to_numeric(out["close"], errors="coerce"),
                "Volume": pd.to_numeric(out["tick_volume"], errors="coerce"),
            }
        ).dropna(subset=["time", "Open", "High", "Low", "Close"])

        return out.reset_index(drop=True)

    @staticmethod
    def _bar_to_dict(bar: Any) -> dict[str, Any]:
        if isinstance(bar, dict):
            return dict(bar)

        if hasattr(bar, "_asdict"):
            as_dict = bar._asdict()
            if isinstance(as_dict, dict):
                return dict(as_dict)

        dtype = getattr(bar, "dtype", None)
        names = getattr(dtype, "names", None)
        if names:
            return {name: bar[name] for name in names}

        if hasattr(bar, "to_dict"):
            as_dict = bar.to_dict()
            if isinstance(as_dict, dict):
                return dict(as_dict)

        raise ValueError(
            "Unsupported candle row type received from data feed: "
            f"{type(bar).__name__}. Expected dict/namedtuple/structured row."
        )

    def _sync_position_side(self) -> PositionSide:
        positions = self.mt5.positions_get(symbol=self.symbol)
        if positions is None:
            raise RuntimeError(f"positions_get failed during sync, last_error={self.mt5.last_error()}")

        if len(positions) == 0:
            self.position_side = "flat"
            return self.position_side

        if len(positions) > 1:
            raise RuntimeError(
                f"Found {len(positions)} open positions for {self.symbol}. "
                "Aborting as requested (single-position mode)."
            )

        self.position_side = self._position_type_to_side(int(positions[0].type))
        return self.position_side

    def _position_type_to_side(self, position_type: int) -> PositionSide:
        if position_type == self._position_type_buy:
            return "long"
        if position_type == self._position_type_sell:
            return "short"
        raise RuntimeError(f"Unknown MT5 position type: {position_type}")

    def _open_or_reverse(self, target_side: PositionSide, signal_close: float, event_time: int) -> None:
        if target_side not in ("long", "short"):
            self._log("ERROR", "order.submit", error_code="invalid_target_side", side=target_side)
            return

        self._sync_position_side()
        from_side = self.position_side
        if self.position_side == target_side:
            return

        if self.position_side != "flat":
            self._close_single_position(reason=f"reverse_to_{target_side}", event_time=event_time)
            self._sync_position_side()
            if self.position_side != "flat":
                self._log(
                    "WARN",
                    "position.change",
                    error_code="reverse_blocked_close_failed",
                    from_side=from_side,
                    side_now=self.position_side,
                    target_side=target_side,
                )
                return

        sl_price: float | None = None
        if self.cfg.use_hard_stop_loss:
            if target_side == "long":
                sl_price = signal_close * (1.0 - self.cfg.hard_stop_loss_pct)
            else:
                sl_price = signal_close * (1.0 + self.cfg.hard_stop_loss_pct)

        entry_volume = self._entry_volume_from_equity(signal_close=signal_close)
        if entry_volume is None:
            return

        ok = self._send_market_order(
            side=target_side,
            volume=float(entry_volume),
            sl_price=sl_price,
            reason=f"entry_{target_side}",
        )
        if ok:
            self._sync_position_side()
            self._record_event(
                "position_change",
                event_time,
                {"to": self.position_side, "reason": f"entry_{target_side}"},
            )
            bar_date, bar_clock = self._bar_date_time(event_time)
            self._log(
                "INFO",
                "position.change",
                date=bar_date,
                time=bar_clock,
                from_side=from_side,
                to_side=self.position_side,
                reason=f"entry_{target_side}",
            )

    def _entry_volume_from_equity(self, signal_close: float) -> float | None:
        account_info = self.mt5.account_info()
        if account_info is None:
            error_code, error_message = self._mt5_error_fields()
            self._log(
                "ERROR",
                "order.size",
                error_code="account_info_failed",
                mt5_error_code=error_code,
                message=error_message,
            )
            return None

        symbol_info = self.mt5.symbol_info(self.symbol)
        if symbol_info is None:
            error_code, error_message = self._mt5_error_fields()
            self._log(
                "ERROR",
                "order.size",
                error_code="symbol_info_failed",
                mt5_error_code=error_code,
                message=error_message,
                symbol=self.symbol,
            )
            return None

        equity = float(getattr(account_info, "equity", 0.0))
        if not np.isfinite(equity) or equity <= 0.0:
            self._log("ERROR", "order.size", error_code="invalid_equity", equity=equity)
            return None

        close_price = float(signal_close)
        if not np.isfinite(close_price) or close_price <= 0.0:
            self._log("ERROR", "order.size", error_code="invalid_signal_close", signal_close=signal_close)
            return None

        contract_size = float(getattr(symbol_info, "trade_contract_size", 1.0))
        if not np.isfinite(contract_size) or contract_size <= 0.0:
            contract_size = 1.0

        target_notional = equity * float(self.cfg.position_size)
        raw_volume = target_notional / (close_price * contract_size)

        volume_min = float(getattr(symbol_info, "volume_min", 0.0) or 0.0)
        volume_max = float(getattr(symbol_info, "volume_max", 0.0) or 0.0)
        volume_step = float(getattr(symbol_info, "volume_step", 0.0) or 0.0)

        volume = raw_volume
        if volume_step > 0.0:
            base = volume_min if volume_min > 0.0 else 0.0
            if volume <= base:
                volume = base
            else:
                steps = math.floor((volume - base) / volume_step)
                volume = base + steps * volume_step

        if volume_min > 0.0 and volume < volume_min:
            volume = volume_min
        if volume_max > 0.0 and volume > volume_max:
            volume = volume_max

        if volume_step > 0.0:
            step_text = f"{volume_step:.10f}".rstrip("0")
            precision = len(step_text.split(".")[1]) if "." in step_text else 0
            volume = round(volume, precision)

        if not np.isfinite(volume) or volume <= 0.0:
            self._log(
                "ERROR",
                "order.size",
                error_code="computed_volume_invalid",
                equity=equity,
                position_size=float(self.cfg.position_size),
                signal_close=close_price,
                contract_size=contract_size,
                raw_volume=raw_volume,
                volume=volume,
            )
            return None

        self._log(
            "INFO",
            "order.size",
            equity=equity,
            position_size=float(self.cfg.position_size),
            target_notional=target_notional,
            signal_close=close_price,
            contract_size=contract_size,
            volume=volume,
        )
        return float(volume)

    def _close_single_position(self, reason: str, event_time: int) -> bool:
        positions = self.mt5.positions_get(symbol=self.symbol)
        if positions is None:
            error_code, error_message = self._mt5_error_fields()
            self._log(
                "ERROR",
                "position.close",
                error_code="positions_get_failed",
                mt5_error_code=error_code,
                message=error_message,
                reason=reason,
            )
            return False

        if len(positions) == 0:
            self.position_side = "flat"
            return True

        if len(positions) > 1:
            raise RuntimeError(
                f"Found {len(positions)} open positions for {self.symbol}. "
                "Aborting as requested (single-position mode)."
            )

        pos = positions[0]
        if int(pos.type) == self._position_type_buy:
            close_side: PositionSide = "short"
            from_side: PositionSide = "long"
        elif int(pos.type) == self._position_type_sell:
            close_side = "long"
            from_side = "short"
        else:
            raise RuntimeError(f"Unknown position type while closing: {pos.type}")

        ok = self._send_market_order(
            side=close_side,
            volume=float(pos.volume),
            position_ticket=int(pos.ticket),
            reason=reason,
        )
        if ok:
            self.position_side = "flat"
            self._record_event(
                "position_change",
                event_time,
                {"to": "flat", "reason": reason},
            )
            bar_date, bar_clock = self._bar_date_time(event_time)
            self._log(
                "INFO",
                "position.change",
                date=bar_date,
                time=bar_clock,
                from_side=from_side,
                to_side="flat",
                reason=reason,
            )
        return ok

    def _terminal_trade_enabled(self, reason: str) -> bool:
        terminal_info_fn = getattr(self.mt5, "terminal_info", None)
        if not callable(terminal_info_fn):
            return True

        terminal_info = terminal_info_fn()
        if terminal_info is None:
            error_code, error_message = self._mt5_error_fields()
            self._log(
                "WARN",
                "order.guard",
                error_code="terminal_info_failed",
                mt5_error_code=error_code,
                message=error_message,
                reason=reason or "n/a",
            )
            return True

        trade_allowed = bool(getattr(terminal_info, "trade_allowed", True))
        tradeapi_disabled = bool(getattr(terminal_info, "tradeapi_disabled", False))
        if trade_allowed and not tradeapi_disabled:
            return True

        self._log(
            "ERROR",
            "order.guard",
            error_code="autotrading_disabled",
            trade_allowed=int(trade_allowed),
            tradeapi_disabled=int(tradeapi_disabled),
            reason=reason or "n/a",
            hint="Enable MT5 AutoTrading in toolbar and Expert Advisors options",
        )
        return False

    def _send_market_order(
        self,
        side: PositionSide,
        volume: float,
        position_ticket: int | None = None,
        sl_price: float | None = None,
        reason: str = "",
    ) -> bool:
        if not self._terminal_trade_enabled(reason=reason):
            return False

        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            error_code, error_message = self._mt5_error_fields()
            self._log(
                "ERROR",
                "order.submit",
                error_code="symbol_info_tick_failed",
                mt5_error_code=error_code,
                message=error_message,
                side=side,
                reason=reason,
            )
            return False

        if side == "long":
            order_type = self._order_type_buy
            price = float(tick.ask)
        elif side == "short":
            order_type = self._order_type_sell
            price = float(tick.bid)
        else:
            self._log("ERROR", "order.submit", error_code="invalid_side", side=side, reason=reason)
            return False

        request: dict[str, Any] = {
            "action": self._trade_action_deal,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": int(self.cfg.order_deviation),
            "magic": int(self.cfg.magic),
            "comment": f"sama_live:{reason}",
            "type_time": self._order_time_gtc,
            "type_filling": self._order_filling_ioc,
        }
        if position_ticket is not None:
            request["position"] = int(position_ticket)
        if sl_price is not None and sl_price > 0.0:
            request["sl"] = float(sl_price)

        log_fields: dict[str, Any] = {
            "side": side,
            "volume": float(volume),
            "price": price,
            "reason": reason or "n/a",
        }
        if position_ticket is not None:
            log_fields["position_ticket"] = int(position_ticket)
        if sl_price is not None and sl_price > 0.0:
            log_fields["sl"] = float(sl_price)
        self._log("INFO", "order.submit", **log_fields)

        result = self.mt5.order_send(request)
        if result is None:
            error_code, error_message = self._mt5_error_fields()
            self._log(
                "ERROR",
                "order.result",
                status="none",
                error_code="order_send_none",
                mt5_error_code=error_code,
                message=error_message,
                reason=reason or "n/a",
            )
            return False

        retcode = int(getattr(result, "retcode", -1))
        if retcode == self._retcode_client_disable:
            self._log(
                "ERROR",
                "order.result",
                status="rejected",
                error_code="autotrading_disabled",
                retcode=retcode,
                ticket=getattr(result, "order", "n/a"),
                comment=getattr(result, "comment", ""),
                reason=reason or "n/a",
                hint="Enable MT5 AutoTrading in toolbar and Expert Advisors options",
            )
            return False

        if retcode not in (self._retcode_done, self._retcode_placed):
            self._log(
                "WARN",
                "order.result",
                status="rejected",
                error_code="retcode_rejected",
                retcode=retcode,
                ticket=getattr(result, "order", "n/a"),
                comment=getattr(result, "comment", ""),
                reason=reason or "n/a",
            )
            return False

        self._log(
            "INFO",
            "order.result",
            status="accepted",
            retcode=retcode,
            ticket=getattr(result, "order", "n/a"),
            comment=getattr(result, "comment", ""),
            reason=reason or "n/a",
        )
        self._record_trade(
            side=side,
            volume=float(volume),
            requested_price=price,
            result=result,
            reason=reason,
            position_ticket=position_ticket,
            sl_price=sl_price,
        )
        return True

    def _record_event(self, event: str, bar_time: int, payload: dict[str, Any]) -> None:
        self._events.append({"time": int(bar_time), "event": event, **payload})

    def _record_trade(
        self,
        side: PositionSide,
        volume: float,
        requested_price: float,
        result: Any,
        reason: str,
        position_ticket: int | None,
        sl_price: float | None,
    ) -> None:
        trade: dict[str, Any] = {
            "logged_at": self._utc_now(),
            "symbol": self.symbol,
            "side": side,
            "volume": float(volume),
            "requested_price": float(requested_price),
            "filled_price": float(getattr(result, "price", requested_price)),
            "retcode": int(getattr(result, "retcode", -1)),
            "ticket": int(getattr(result, "order", 0)),
            "comment": str(getattr(result, "comment", "")),
            "reason": reason or "n/a",
        }
        if position_ticket is not None:
            trade["position_ticket"] = int(position_ticket)
        if sl_price is not None and sl_price > 0.0:
            trade["sl"] = float(sl_price)
        self._trades.append(trade)

    def _mt5_error_fields(self) -> tuple[str, str]:
        err = self.mt5.last_error()
        if isinstance(err, tuple) and len(err) >= 2:
            return str(err[0]), str(err[1])
        return "unknown", str(err)

    def _log(self, level: str, event: str, **fields: Any) -> None:
        parts: list[str] = []
        for key, value in fields.items():
            text = str(value)
            if " " in text:
                text = f'"{text}"'
            parts.append(f"{key}={text}")

        line = f"{self._utc_now()} | {level.upper()} | strategy | {event}"
        if parts:
            line += f" | {' '.join(parts)}"
        print(line)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _bar_date_time(ts: int) -> tuple[str, str]:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
