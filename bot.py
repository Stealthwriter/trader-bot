"""
Simple ETH polling bot for MetaTrader 5.

It initializes MT5, preloads the last 500 closed H1 candles, and feeds
newly closed bars to the strategy. Strategy handles all order management.
"""

from __future__ import annotations

from collections import deque
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mt5linux import MetaTrader5
from strategy import SamaLiveStrategy


H1_SECONDS = 3600
CONFIG_PATH = Path("bot_config.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _bar_date_time(bar_time: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(int(bar_time), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")


def _mt5_error_fields(mt5: MetaTrader5) -> tuple[str, str]:
    err = mt5.last_error()
    if isinstance(err, tuple) and len(err) >= 2:
        return str(err[0]), str(err[1])
    return "unknown", str(err)


def _log(level: str, component: str, event: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{_utc_now()} | {level.upper()} | {component} | {event}"
    if payload:
        line += f" | {payload}"
    print(line)


def load_bot_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Config root must be a JSON object.")

    mt5_cfg = data.get("mt5")
    if not isinstance(mt5_cfg, dict):
        raise TypeError("Config key 'mt5' must be a JSON object.")
    if "host" not in mt5_cfg or "port" not in mt5_cfg:
        raise ValueError("Config key 'mt5' must contain 'host' and 'port'.")

    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise TypeError("Config key 'symbol' must be a non-empty string.")

    strategy_cfg = data.get("strategy")
    if not isinstance(strategy_cfg, dict):
        raise TypeError("Config key 'strategy' must be a JSON object.")

    return data


def main() -> int:
    bot_cfg = load_bot_config(CONFIG_PATH)

    mt5_cfg = bot_cfg["mt5"]
    mt5_host = str(mt5_cfg["host"])
    mt5_port = int(mt5_cfg["port"])
    mt5_timeout_ms = int(mt5_cfg.get("timeout_ms", 120000))

    symbol = str(bot_cfg["symbol"])
    poll_seconds = float(bot_cfg.get("poll_seconds", 1.0))
    lookback_candles = int(bot_cfg.get("lookback_candles", 500))
    if lookback_candles <= 0:
        raise ValueError("Config key 'lookback_candles' must be > 0.")

    print_replay_events = bool(bot_cfg.get("print_replay_events", False))
    print_replay_events_limit = int(bot_cfg.get("print_replay_events_limit", 20))
    strategy_cfg = bot_cfg["strategy"]

    mt5 = MetaTrader5(host=mt5_host, port=mt5_port)

    if not mt5.initialize(timeout=mt5_timeout_ms):
        error_code, error_message = _mt5_error_fields(mt5)
        _log(
            "ERROR",
            "bot",
            "startup.init",
            error_code=error_code,
            context="initialize",
            message=error_message,
        )
        return 1

    if not mt5.symbol_select(symbol, True):
        error_code, error_message = _mt5_error_fields(mt5)
        _log(
            "ERROR",
            "bot",
            "startup.symbol_select",
            error_code=error_code,
            symbol=symbol,
            message=error_message,
        )
        mt5.shutdown()
        return 1

    try:
        initial_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, lookback_candles)
        if initial_rates is None or len(initial_rates) == 0:
            error_code, error_message = _mt5_error_fields(mt5)
            _log(
                "ERROR",
                "bot",
                "startup.load_initial",
                error_code=error_code,
                symbol=symbol,
                message=error_message,
            )
            return 1

        strategy = SamaLiveStrategy.from_config(mt5=mt5, symbol=symbol, config_data=strategy_cfg)

        candles = deque(initial_rates, maxlen=lookback_candles)
        last_closed_time = int(candles[-1]["time"])
        strategy.initialize(candles)

        if print_replay_events:
            replay = strategy.replay_events(candles)
            print(f"[debug] replay_events_count={len(replay)}")
            for event in replay[-print_replay_events_limit:]:
                ts = datetime.fromtimestamp(int(event["time"]), tz=timezone.utc).isoformat()
                print(f"[debug] {ts} {event}")

        _log(
            "INFO",
            "bot",
            "startup.ready",
            symbol=symbol,
            timeframe="H1",
            loaded_bars=len(candles),
            poll_seconds=f"{poll_seconds:g}",
        )
        initial_date, initial_time = _bar_date_time(last_closed_time)
        _log(
            "INFO",
            "bot",
            "candle.last",
            date=initial_date,
            time=initial_time,
            close=float(candles[-1]["close"]),
        )

        while True:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, 1)

            if rates is None or len(rates) == 0:
                error_code, error_message = _mt5_error_fields(mt5)
                _log(
                    "WARN",
                    "bot",
                    "data.fetch",
                    error_code=error_code,
                    symbol=symbol,
                    message=error_message,
                )
            else:
                latest_closed = rates[0]
                latest_closed_time = int(latest_closed["time"])

                if latest_closed_time > last_closed_time:
                    bars_missed = max(1, (latest_closed_time - last_closed_time) // H1_SECONDS)
                    missing_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, bars_missed)

                    bars_to_process = [latest_closed]
                    if missing_rates is not None and len(missing_rates) > 0:
                        bars_to_process = sorted(missing_rates, key=lambda b: int(b["time"]))

                    for bar in bars_to_process:
                        bar_time = int(bar["time"])
                        if bar_time <= last_closed_time:
                            continue

                        candles.append(bar)
                        last_closed_time = bar_time
                        bar_date, bar_clock = _bar_date_time(bar_time)
                        _log(
                            "INFO",
                            "bot",
                            "candle.closed",
                            date=bar_date,
                            time=bar_clock,
                            close=float(bar["close"]),
                        )
                        strategy.on_new_closed_bar(candles)

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        _log("INFO", "bot", "shutdown.user_interrupt")
    except Exception as exc:
        _log("ERROR", "bot", "runtime.fatal", error_code="unhandled_exception", message=exc)
        return 1
    finally:
        mt5.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
