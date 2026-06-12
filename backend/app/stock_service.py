from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any

_SINA_QUOTE_URL = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
_SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "application/json, text/plain, */*",
}
_SINA_PAGE_SIZE = 200
_SINA_MAX_PAGES = 32
_CACHE_TTL_SECONDS = 180
_CACHE: dict[str, Any] = {"expires_at": 0.0, "items": None}


def _clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _scale(value: float | None, lower: float, upper: float) -> float:
    if value is None or upper <= lower:
        return 0.0
    return _clamp((value - lower) / (upper - lower) * 10.0)


def _board_label(code: str) -> str:
    if code.startswith("68"):
        return "科创板"
    if code.startswith("30"):
        return "创业板"
    if code.startswith(("4", "8", "9", "92", "83", "87", "88")):
        return "北交所"
    if code.startswith("6"):
        return "沪市主板"
    return "深市主板"


def _format_symbol(raw_symbol: str, code: str) -> str:
    market = raw_symbol[:2].upper() if raw_symbol else "SZ"
    return f"{code}.{market}"


def _fetch_sina_page(page: int) -> list[dict[str, Any]]:
    params = {
        "page": str(page),
        "num": str(_SINA_PAGE_SIZE),
        "sort": "symbol",
        "asc": "1",
        "node": "hs_a",
        "symbol": "",
        "_s_r_a": "page",
    }
    url = f"{_SINA_QUOTE_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=_SINA_HEADERS)
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("gbk", errors="ignore")
    data = json.loads(body)
    return data if isinstance(data, list) else []


def _fetch_realtime_snapshot() -> list[dict[str, Any]]:
    now = time.monotonic()
    cached_items = _CACHE.get("items")
    if cached_items and float(_CACHE.get("expires_at", 0.0)) > now:
        return cached_items

    merged_items: list[dict[str, Any]] = []
    for page in range(1, _SINA_MAX_PAGES + 1):
        page_items = _fetch_sina_page(page)
        if not page_items:
            break
        merged_items.extend(page_items)
        if len(page_items) < _SINA_PAGE_SIZE:
            break

    if not merged_items:
        if cached_items:
            return cached_items
        raise RuntimeError("未从新浪财经获取到实时股票行情")

    _CACHE["expires_at"] = now + _CACHE_TTL_SECONDS
    _CACHE["items"] = merged_items
    return merged_items


def _is_candidate(item: dict[str, Any]) -> bool:
    name = str(item.get("name") or "")
    latest_price = _safe_float(item.get("trade"))
    total_market_cap = _safe_float(item.get("mktcap"))
    turnover_rate = _safe_float(item.get("turnoverratio"))
    amount = _safe_float(item.get("amount"))

    if not name or latest_price is None or latest_price <= 0:
        return False
    if total_market_cap is None or total_market_cap < 300000:
        return False
    if amount is None or amount < 80_000_000:
        return False
    if turnover_rate is not None and turnover_rate > 35:
        return False
    if "ST" in name.upper() or "退" in name or name.startswith(("N", "C")):
        return False
    return True


def _build_board_heat(items: list[dict[str, Any]]) -> dict[str, float]:
    board_buckets: dict[str, list[dict[str, float]]] = {}
    for item in items:
        code = str(item.get("code") or "")
        board = _board_label(code)
        board_buckets.setdefault(board, []).append(
            {
                "change_percent": _safe_float(item.get("changepercent")) or 0.0,
                "turnover_rate": _safe_float(item.get("turnoverratio")) or 0.0,
                "amount": _safe_float(item.get("amount")) or 0.0,
            }
        )

    heat_map: dict[str, float] = {}
    for board, values in board_buckets.items():
        avg_change = sum(entry["change_percent"] for entry in values) / max(len(values), 1)
        avg_turnover = sum(entry["turnover_rate"] for entry in values) / max(len(values), 1)
        avg_amount = sum(entry["amount"] for entry in values) / max(len(values), 1)
        heat = (
            _scale(avg_change, -2.0, 5.0) * 0.5
            + _scale(avg_turnover, 0.8, 8.0) * 0.2
            + _scale(math.log10(max(avg_amount, 1.0)), 8.0, 10.5) * 0.3
        )
        heat_map[board] = round(_clamp(heat), 2)
    return heat_map


def _valuation_score(pe_dynamic: float | None, pb_ratio: float | None) -> float:
    if pe_dynamic is None or pe_dynamic <= 0:
        pe_score = 2.0
    elif pe_dynamic <= 15:
        pe_score = 9.2
    elif pe_dynamic <= 30:
        pe_score = 8.3
    elif pe_dynamic <= 45:
        pe_score = 6.9
    elif pe_dynamic <= 80:
        pe_score = 5.1
    else:
        pe_score = 3.6

    if pb_ratio is None or pb_ratio <= 0:
        pb_score = 4.0
    elif pb_ratio <= 2:
        pb_score = 9.0
    elif pb_ratio <= 4:
        pb_score = 7.8
    elif pb_ratio <= 7:
        pb_score = 6.1
    else:
        pb_score = 4.4

    return round(_clamp(pe_score * 0.65 + pb_score * 0.35), 1)


def _amplitude(high: float | None, low: float | None, previous_close: float | None) -> float:
    if high is None or low is None or previous_close is None or previous_close <= 0:
        return 0.0
    return round((high - low) / previous_close * 100, 2)


def _style_label(score: float, risk_score: float, change_percent: float) -> str:
    if score >= 84 and risk_score <= 4.3:
        return "强趋势低回撤"
    if score >= 81:
        return "高景气进攻"
    if risk_score <= 3.8:
        return "稳健增强"
    if change_percent >= 4:
        return "动量强化"
    return "均衡配置"


def _build_reasons(board: str, metrics: dict[str, float], item: dict[str, Any]) -> list[str]:
    reasons = [
        f"{board} 热度分 {metrics['catalyst']}，当前板块风格仍有承接力度",
        f"趋势分 {metrics['trend']}，当日涨跌幅与开盘后强度表现靠前",
        f"量能分 {metrics['volume']}，换手 {metrics['turnover_rate']}% 与成交额共同验证关注度",
    ]
    if metrics["valuation"] >= 7.0:
        reasons.append(f"估值得分 {metrics['valuation']}，动态 PE / PB 处于相对可接受区间")
    return reasons[:4]


def _build_risks(metrics: dict[str, float], item: dict[str, Any], amplitude: float) -> list[str]:
    risks: list[str] = []
    turnover_rate = _safe_float(item.get("turnoverratio")) or 0.0
    pe_dynamic = _safe_float(item.get("per"))
    change_percent = _safe_float(item.get("changepercent")) or 0.0

    if metrics["risk"] >= 6.8:
        risks.append("波动分偏高，建议严格控制仓位与止损纪律")
    if amplitude >= 12:
        risks.append(f"当日振幅 {amplitude}% 较大，短线分歧可能进一步放大")
    if turnover_rate >= 18:
        risks.append(f"换手率 {round(turnover_rate, 2)}% 偏高，需警惕情绪过热")
    if pe_dynamic is None or pe_dynamic <= 0:
        risks.append("动态 PE 为负或不可用，盈利质量需要额外甄别")
    if change_percent >= 9:
        risks.append("当日涨幅较大，追高时需关注次日承接强度")
    return risks[:3] or [f"当前风险分 {metrics['risk']}，建议结合仓位管理持续跟踪"]


def _analyze_stock(item: dict[str, Any], board_heat: dict[str, float]) -> dict[str, Any]:
    raw_symbol = str(item.get("symbol") or "")
    code = str(item.get("code") or raw_symbol[-6:])
    board = _board_label(code)

    latest_price = _safe_float(item.get("trade")) or 0.0
    previous_close = _safe_float(item.get("settlement"))
    open_price = _safe_float(item.get("open"))
    high_price = _safe_float(item.get("high"))
    low_price = _safe_float(item.get("low"))
    change_amount = _safe_float(item.get("pricechange")) or 0.0
    change_percent = _safe_float(item.get("changepercent")) or 0.0
    turnover_rate = _safe_float(item.get("turnoverratio")) or 0.0
    amount = _safe_float(item.get("amount")) or 0.0
    volume = _safe_float(item.get("volume")) or 0.0
    pe_dynamic = _safe_float(item.get("per"))
    pb_ratio = _safe_float(item.get("pb"))
    total_market_cap = (_safe_float(item.get("mktcap")) or 0.0) * 10000
    circulating_market_cap = (_safe_float(item.get("nmc")) or 0.0) * 10000
    amplitude = _amplitude(high_price, low_price, previous_close)
    intraday_strength = None
    if open_price and open_price > 0:
        intraday_strength = (latest_price - open_price) / open_price * 100

    trend = _clamp(
        _scale(change_percent, -3.0, 9.0) * 0.48
        + _scale(intraday_strength, -2.0, 7.0) * 0.32
        + _scale((latest_price - (previous_close or latest_price)) / max(previous_close or 1.0, 1.0) * 100, -3.0, 9.0) * 0.2
    )
    valuation = _valuation_score(pe_dynamic, pb_ratio)
    volume_score = _clamp(
        _scale(turnover_rate, 1.0, 15.0) * 0.4
        + _scale(math.log10(max(amount, 1.0)), 8.0, 10.5) * 0.4
        + _scale(math.log10(max(volume, 1.0)), 6.0, 8.8) * 0.2
    )
    quality = _clamp(
        _scale(math.log10(max(total_market_cap, 1.0)), 9.5, 12.2) * 0.4
        + (7.8 if pe_dynamic is not None and pe_dynamic > 0 else 2.8) * 0.35
        + (10.0 - _scale(amplitude, 2.0, 18.0)) * 0.25
    )
    catalyst = _clamp(
        board_heat.get(board, 5.0) * 0.45
        + _scale(change_percent, -2.0, 8.0) * 0.25
        + _scale(turnover_rate, 1.0, 15.0) * 0.15
        + _scale(math.log10(max(amount, 1.0)), 8.0, 10.5) * 0.15
    )
    risk = _clamp(
        _scale(amplitude, 4.0, 20.0) * 0.35
        + _scale(turnover_rate, 10.0, 30.0) * 0.25
        + (7.8 if pe_dynamic is None or pe_dynamic <= 0 else 2.6) * 0.2
        + (10.0 - _scale(math.log10(max(circulating_market_cap, 1.0)), 9.3, 11.8)) * 0.2
    )
    confidence = _clamp(
        quality * 0.29
        + trend * 0.25
        + volume_score * 0.18
        + valuation * 0.14
        + catalyst * 0.14
        - risk * 0.08
    )
    final_score = round(
        _clamp(
            trend * 0.29
            + quality * 0.19
            + volume_score * 0.16
            + valuation * 0.12
            + catalyst * 0.24
            - risk * 0.1
            + 1.1
        )
        * 10,
        1,
    )

    metrics = {
        "trend": round(trend, 1),
        "quality": round(quality, 1),
        "volume": round(volume_score, 1),
        "valuation": round(valuation, 1),
        "catalyst": round(catalyst, 1),
        "risk": round(risk, 1),
        "confidence": round(confidence, 1),
        "turnover_rate": round(turnover_rate, 2),
    }

    return {
        "symbol": _format_symbol(raw_symbol, code),
        "name": str(item.get("name") or code),
        "sector": board,
        "board": board,
        "score": final_score,
        "style": _style_label(final_score, metrics["risk"], change_percent),
        "latest_price": round(latest_price, 2),
        "change_percent": round(change_percent, 2),
        "change_amount": round(change_amount, 2),
        "turnover_rate": round(turnover_rate, 2),
        "amplitude": round(amplitude, 2),
        "pe_dynamic": round(pe_dynamic, 2) if pe_dynamic is not None else None,
        "pb_ratio": round(pb_ratio, 2) if pb_ratio is not None else None,
        "total_market_cap": round(total_market_cap, 2),
        "circulating_market_cap": round(circulating_market_cap, 2),
        "amount": round(amount, 2),
        "volume": round(volume, 2),
        "open_price": round(open_price, 2) if open_price is not None else None,
        "high_price": round(high_price, 2) if high_price is not None else None,
        "low_price": round(low_price, 2) if low_price is not None else None,
        "previous_close": round(previous_close, 2) if previous_close is not None else None,
        "reasons": _build_reasons(board, metrics, item),
        "risk_flags": _build_risks(metrics, item, round(amplitude, 2)),
        "metrics": metrics,
    }


def _market_temperature(all_items: list[dict[str, Any]]) -> tuple[float, int, int, float]:
    changes = [(_safe_float(item.get("changepercent")) or 0.0) for item in all_items if _safe_float(item.get("trade")) is not None]
    rising = sum(1 for change in changes if change > 0)
    falling = sum(1 for change in changes if change < 0)
    avg_change = sum(changes) / max(len(changes), 1)
    rise_ratio = rising / max(len(changes), 1)
    temperature = _clamp(3.8 + rise_ratio * 4.2 + avg_change * 0.22)
    return round(temperature, 1), rising, falling, round(rise_ratio * 100, 1)


def _market_view(trading_day: date, all_items: list[dict[str, Any]], picks: list[dict[str, Any]], board_heat: dict[str, float], candidate_size: int) -> dict[str, Any]:
    temperature, up_count, down_count, rising_ratio = _market_temperature(all_items)
    avg_score = round(sum(item["score"] for item in picks) / max(len(picks), 1), 1)
    hot_sectors = sorted({item["sector"] for item in picks}, key=lambda board: (-board_heat.get(board, 0.0), board))[:3]

    style = "偏成长进攻"
    if temperature < 5.5:
        style = "偏防守均衡"
    elif temperature < 6.6:
        style = "均衡偏成长"

    return {
        "trading_day": trading_day.isoformat(),
        "market_temperature": temperature,
        "average_score": avg_score,
        "style": style,
        "hot_sectors": hot_sectors,
        "summary": f"全市场上涨家数 {up_count}，下跌家数 {down_count}，上涨占比 {rising_ratio}% 。当前更容易跑出强势股的方向集中在 {'、'.join(hot_sectors) if hot_sectors else '主板与成长风格'}。",
        "up_count": up_count,
        "down_count": down_count,
        "rising_ratio": rising_ratio,
        "universe_size": len(all_items),
        "candidate_size": candidate_size,
    }


def get_daily_stock_picks(limit: int = 10, trading_day: date | None = None) -> dict[str, Any]:
    trading_day = trading_day or date.today()
    snapshot_items = _fetch_realtime_snapshot()
    candidates = [item for item in snapshot_items if _is_candidate(item)]
    if not candidates:
        raise RuntimeError("实时行情已返回，但当前没有可用于分析的候选股票")

    board_heat = _build_board_heat(candidates)
    ranked = [_analyze_stock(item, board_heat) for item in candidates]
    ranked.sort(
        key=lambda item: (
            -item["score"],
            -item["metrics"]["confidence"],
            -item["change_percent"],
            item["symbol"],
        )
    )

    top = [{"rank": index, **item} for index, item in enumerate(ranked[: min(limit, 10)], start=1)]
    market_view = _market_view(trading_day, snapshot_items, top, board_heat, len(candidates))

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trading_day": trading_day.isoformat(),
        "title": f"{trading_day.isoformat()} 实时潜力股 Top {len(top)}",
        "summary": "基于新浪财经实时 A 股行情，对趋势、量能、估值、板块热度和波动风险进行综合打分。",
        "methodology": [
            "趋势强度：使用当日涨跌幅、开盘后强度和收盘位置衡量动量延续性",
            "量能确认：使用换手率、成交额、成交量衡量资金活跃度",
            "估值约束：使用动态 PE、PB 对安全边际进行打分",
            "板块热度：结合主板 / 创业板 / 科创板 / 北交所风格强弱识别主线",
            "风险惩罚：对高振幅、高换手、负 PE 和小流通盘进行惩罚",
        ],
        "market_view": market_view,
        "picks": top,
        "disclaimer": "数据来自新浪财经公开行情接口，结果仅用于研究、演示和策略观察，不构成任何投资建议。",
        "data_source": "新浪财经实时行情",
    }
