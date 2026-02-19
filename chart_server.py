from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from mt5linux import MetaTrader5
import numpy as np
import pandas as pd  # type: ignore[reportMissingImports]
from pydantic import BaseModel

from strategy import build_signal_frame


H1_SECONDS = 3600
DEFAULT_LIMIT = 200
MAX_LIMIT = 2000
POLL_INTERVAL_SECONDS = 5.0

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
BOT_CONFIG_FILE = BASE_DIR / "bot_config.json"

LIVE_CANDLES_LOCK = threading.Lock()
LIVE_CANDLES: list["Candle"] = []

STOP_EVENT = threading.Event()
POLLER_THREAD: threading.Thread | None = None
MT5_CLIENT: MetaTrader5 | None = None
LAST_CLOSED_TIME: int | None = None


class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class IndicatorConfig:
    ama_length: int
    major_length: int
    minor_length: int
    slope_period: int
    slope_in_range: float
    flat_threshold: float


@dataclass(frozen=True)
class ServerConfig:
    mt5_host: str
    mt5_port: int
    mt5_timeout_ms: int
    symbol: str
    lookback_candles: int
    indicator: IndicatorConfig


def _log(level: str, event: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{level.upper()} | chart_server | {event}"
    if payload:
        line += f" | {payload}"
    print(line)


def _load_server_config(path: Path) -> ServerConfig:
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("bot_config.json root must be a JSON object.")

    mt5_cfg = data.get("mt5")
    if not isinstance(mt5_cfg, dict):
        raise RuntimeError("bot_config.json missing object key: mt5")

    strategy_cfg = data.get("strategy")
    if not isinstance(strategy_cfg, dict):
        raise RuntimeError("bot_config.json missing object key: strategy")

    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise RuntimeError("bot_config.json key 'symbol' must be a non-empty string.")

    lookback_candles = int(data.get("lookback_candles", 500))
    if lookback_candles <= 0:
        raise RuntimeError("bot_config.json key 'lookback_candles' must be > 0.")

    indicator_cfg = IndicatorConfig(
        ama_length=int(strategy_cfg["ama_length"]),
        major_length=int(strategy_cfg["major_length"]),
        minor_length=int(strategy_cfg["minor_length"]),
        slope_period=int(strategy_cfg["slope_period"]),
        slope_in_range=float(strategy_cfg["slope_in_range"]),
        flat_threshold=float(strategy_cfg["flat_threshold"]),
    )

    return ServerConfig(
        mt5_host=str(mt5_cfg["host"]),
        mt5_port=int(mt5_cfg["port"]),
        mt5_timeout_ms=int(mt5_cfg.get("timeout_ms", 120000)),
        symbol=symbol.strip(),
        lookback_candles=lookback_candles,
        indicator=indicator_cfg,
    )


SERVER_CFG = _load_server_config(BOT_CONFIG_FILE)


def _mt5_error_fields(mt5: MetaTrader5) -> tuple[str, str]:
    err = mt5.last_error()
    if isinstance(err, tuple) and len(err) >= 2:
        return str(err[0]), str(err[1])
    return "unknown", str(err)


def _rate_to_candle(rate: Any) -> Candle:
    return Candle(
        time=int(rate["time"]),
        open=float(rate["open"]),
        high=float(rate["high"]),
        low=float(rate["low"]),
        close=float(rate["close"]),
    )


def _connect_and_preload() -> None:
    global MT5_CLIENT, LAST_CLOSED_TIME
    mt5 = MetaTrader5(host=SERVER_CFG.mt5_host, port=SERVER_CFG.mt5_port)

    if not mt5.initialize(timeout=SERVER_CFG.mt5_timeout_ms):
        error_code, error_message = _mt5_error_fields(mt5)
        raise RuntimeError(f"MT5 initialize failed: {error_code} {error_message}")

    if not mt5.symbol_select(SERVER_CFG.symbol, True):
        error_code, error_message = _mt5_error_fields(mt5)
        mt5.shutdown()
        raise RuntimeError(f"MT5 symbol_select failed: {error_code} {error_message}")

    rates = mt5.copy_rates_from_pos(
        SERVER_CFG.symbol,
        mt5.TIMEFRAME_H1,
        1,
        SERVER_CFG.lookback_candles,
    )
    if rates is None or len(rates) == 0:
        error_code, error_message = _mt5_error_fields(mt5)
        mt5.shutdown()
        raise RuntimeError(f"MT5 initial preload failed: {error_code} {error_message}")

    candles = sorted((_rate_to_candle(rate) for rate in rates), key=lambda c: c.time)
    with LIVE_CANDLES_LOCK:
        LIVE_CANDLES.clear()
        LIVE_CANDLES.extend(candles[-SERVER_CFG.lookback_candles :])
        LAST_CLOSED_TIME = LIVE_CANDLES[-1].time

    MT5_CLIENT = mt5
    _log(
        "INFO",
        "startup.ready",
        symbol=SERVER_CFG.symbol,
        preload_bars=len(LIVE_CANDLES),
        last_closed_time=LAST_CLOSED_TIME,
    )


def _poll_loop() -> None:
    global LAST_CLOSED_TIME
    while not STOP_EVENT.wait(POLL_INTERVAL_SECONDS):
        mt5 = MT5_CLIENT
        if mt5 is None:
            continue
        try:
            rates = mt5.copy_rates_from_pos(SERVER_CFG.symbol, mt5.TIMEFRAME_H1, 1, 1)
            if rates is None or len(rates) == 0:
                error_code, error_message = _mt5_error_fields(mt5)
                _log("WARN", "poll.fetch_latest", error_code=error_code, message=error_message)
                continue

            latest = rates[0]
            latest_time = int(latest["time"])
            if LAST_CLOSED_TIME is not None and latest_time <= LAST_CLOSED_TIME:
                continue

            bars_missed = 1 if LAST_CLOSED_TIME is None else max(1, (latest_time - LAST_CLOSED_TIME) // H1_SECONDS)
            missing_rates = mt5.copy_rates_from_pos(
                SERVER_CFG.symbol,
                mt5.TIMEFRAME_H1,
                1,
                bars_missed,
            )
            bars = [latest]
            if missing_rates is not None and len(missing_rates) > 0:
                bars = sorted(missing_rates, key=lambda b: int(b["time"]))

            with LIVE_CANDLES_LOCK:
                for bar in bars:
                    bar_time = int(bar["time"])
                    if LAST_CLOSED_TIME is not None and bar_time <= LAST_CLOSED_TIME:
                        continue
                    LIVE_CANDLES.append(_rate_to_candle(bar))
                    LAST_CLOSED_TIME = bar_time
                if len(LIVE_CANDLES) > SERVER_CFG.lookback_candles:
                    LIVE_CANDLES[:] = LIVE_CANDLES[-SERVER_CFG.lookback_candles :]
        except Exception as exc:
            _log("WARN", "poll.exception", message=exc)


def _current_candles(limit: int) -> list[Candle]:
    with LIVE_CANDLES_LOCK:
        if not LIVE_CANDLES:
            raise HTTPException(status_code=503, detail="MT5 candle cache is not ready.")
        return list(LIVE_CANDLES[-limit:])


def _to_ohlcv_frame(candles: list[Candle]) -> pd.DataFrame:
    rows = [
        {
            "Open": float(c.open),
            "High": float(c.high),
            "Low": float(c.low),
            "Close": float(c.close),
            "Volume": 0.0,
        }
        for c in candles
    ]
    return pd.DataFrame(rows)


def _sama_line_and_markers(
    candles: list[Candle], frame: pd.DataFrame
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | str]]]:
    sama_line: list[dict[str, int | float | str]] = []
    markers: list[dict[str, int | str]] = []

    for idx, candle in enumerate(candles):
        time_value = int(candle.time)
        sama_value = float(frame["sama"].iloc[idx]) if idx < len(frame) else float("nan")
        slope_value = float(frame["slope"].iloc[idx]) if idx < len(frame) else float("nan")
        long_flip = bool(frame["long_flip"].iloc[idx]) if idx < len(frame) else False
        short_flip = bool(frame["short_flip"].iloc[idx]) if idx < len(frame) else False
        has_sama = np.isfinite(sama_value)

        if has_sama and np.isfinite(slope_value):
            threshold = float(SERVER_CFG.indicator.flat_threshold)
            if slope_value > threshold:
                color = "#22c55e"
            elif slope_value < -threshold:
                color = "#ef4444"
            else:
                color = "#facc15"
            sama_line.append({"time": time_value, "value": sama_value, "color": color})
        elif has_sama:
            sama_line.append({"time": time_value, "value": sama_value, "color": "#facc15"})
        else:
            sama_line.append({"time": time_value})

        if long_flip:
            markers.append(
                {
                    "time": time_value,
                    "position": "belowBar",
                    "shape": "circle",
                    "color": "#22c55e",
                }
            )
        if short_flip:
            markers.append(
                {
                    "time": time_value,
                    "position": "aboveBar",
                    "shape": "circle",
                    "color": "#ef4444",
                }
            )

    return sama_line, markers


app = FastAPI(title="TradingView Lightweight Charts Demo")


@app.on_event("startup")
def on_startup() -> None:
    global POLLER_THREAD
    _connect_and_preload()
    STOP_EVENT.clear()
    POLLER_THREAD = threading.Thread(target=_poll_loop, name="mt5-candle-poller", daemon=True)
    POLLER_THREAD.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    STOP_EVENT.set()
    if POLLER_THREAD is not None:
        POLLER_THREAD.join(timeout=2.0)
    if MT5_CLIENT is not None:
        MT5_CLIENT.shutdown()


@app.get("/", response_class=FileResponse)
def get_index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/api/candles", response_model=list[Candle])
def get_candles(limit: int = Query(default=DEFAULT_LIMIT, ge=10, le=MAX_LIMIT)) -> list[Candle]:
    return _current_candles(limit=limit)


@app.get("/api/indicator")
def get_indicator(limit: int = Query(default=DEFAULT_LIMIT, ge=10, le=MAX_LIMIT)) -> dict[str, object]:
    candles = _current_candles(limit=limit)
    ohlcv = _to_ohlcv_frame(candles)
    frame = build_signal_frame(
        ohlcv=ohlcv,
        ama_length=SERVER_CFG.indicator.ama_length,
        major_length=SERVER_CFG.indicator.major_length,
        minor_length=SERVER_CFG.indicator.minor_length,
        slope_period=SERVER_CFG.indicator.slope_period,
        slope_in_range=SERVER_CFG.indicator.slope_in_range,
        flat_threshold=SERVER_CFG.indicator.flat_threshold,
    )
    sama_line, markers = _sama_line_and_markers(candles, frame)
    return {
        "config": {
            "ama_length": SERVER_CFG.indicator.ama_length,
            "major_length": SERVER_CFG.indicator.major_length,
            "minor_length": SERVER_CFG.indicator.minor_length,
            "slope_period": SERVER_CFG.indicator.slope_period,
            "slope_in_range": SERVER_CFG.indicator.slope_in_range,
            "flat_threshold": SERVER_CFG.indicator.flat_threshold,
        },
        "sama_line": sama_line,
        "markers": markers,
    }
