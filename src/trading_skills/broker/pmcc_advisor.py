# ABOUTME: Analyzes PMCC (Poor Man's Covered Call / diagonal spread) positions from IB portfolio.
# ABOUTME: Advises on short leg risk, daily P&L projections, and roll recommendations.

import asyncio
import math
from collections import defaultdict
from datetime import datetime, timedelta

from scipy.stats import norm

from trading_skills.black_scholes import (
    black_scholes_delta,
    black_scholes_price,
    estimate_iv,
    implied_volatility,
)
from trading_skills.broker.connection import (
    CLIENT_IDS,
    best_option_chain,
    fetch_positions,
    fetch_spot_prices,
    ib_connection,
    normalize_positions,
)
from trading_skills.utils import _NY, days_to_expiry, fetch_with_timeout, generated_at_str

RISK_FREE_RATE = 0.045
NET_CREDIT_MIN = -0.10  # max debit allowed when rolling


# ===========================================================================
# ANALYTICS (no IBKR dependency — fully testable in isolation)
# ===========================================================================


def get_option_price(quote: dict, price_mode: str) -> float | None:
    """Extract option price from a quote dict using the configured price mode."""
    bid = quote.get("bid")
    ask = quote.get("ask")
    last = quote.get("last") or quote.get("lastPrice")

    if price_mode == "last":
        return last if last and last > 0 else None

    # mid (default)
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2
    if bid and bid > 0:
        return bid
    if ask and ask > 0:
        return ask
    return last if last and last > 0 else None


def calc_iv(price: float, spot: float, strike: float, dte_days: int, right: str) -> float | None:
    """Calculate implied volatility from option price via Newton-Raphson / bisection."""
    if not price or price <= 0:
        return None
    T = max(dte_days, 0.5) / 365
    opt_type = "call" if right == "C" else "put"
    return implied_volatility(price, spot, strike, T, RISK_FREE_RATE, opt_type)


def calc_delta(spot: float, strike: float, dte_days: int, iv: float, right: str) -> float:
    """Calculate Black-Scholes delta."""
    T = max(dte_days, 0.5) / 365
    opt_type = "call" if right == "C" else "put"
    return black_scholes_delta(spot, strike, T, RISK_FREE_RATE, iv, opt_type)


def calc_assignment_prob(spot: float, strike: float, dte_days: int, iv: float, right: str) -> float:
    """Calculate probability of assignment (= probability of expiring ITM).

    Uses N(d2) for calls, N(-d2) for puts — the risk-neutral probability that
    the option expires in the money.
    """
    T = max(dte_days, 0.5) / 365
    if T <= 0 or iv <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv**2) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    return norm.cdf(d2) if right == "C" else norm.cdf(-d2)


def calc_bs_price(spot: float, strike: float, dte_days: int, iv: float, right: str) -> float:
    """Calculate Black-Scholes option price."""
    T = max(dte_days, 0) / 365
    opt_type = "call" if right == "C" else "put"
    return black_scholes_price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)


def calc_daily_pnl_table(
    long_strike: float,
    long_dte: int,
    long_cost: float,
    long_iv: float,
    short_strike: float,
    short_dte: int,
    short_premium: float,
    short_iv: float,
    qty: int,
    right: str = "C",
) -> list[dict]:
    """Daily P&L projection from today through short expiry.

    Optimal spot = short_strike (highest P&L achievable without triggering assignment).
    P&L = (long_value - long_cost) + (short_premium - short_current_value), scaled by qty * 100.
    """
    results = []
    today = datetime.now(_NY).date()

    for day_offset in range(short_dte + 1):
        current_date = today + timedelta(days=day_offset)
        short_days_rem = max(short_dte - day_offset, 0)
        long_days_rem = max(long_dte - day_offset, 1)

        S = short_strike  # optimal spot: short just OTM, max P&L without assignment

        long_price = calc_bs_price(S, long_strike, long_days_rem, long_iv, right)
        short_price = calc_bs_price(S, short_strike, short_days_rem, short_iv, right)

        pnl_per_share = (long_price - long_cost) + (short_premium - short_price)
        pnl = round(pnl_per_share * qty * 100, 2)

        results.append(
            {
                "date": current_date.isoformat(),
                "days_to_short_expiry": short_days_rem,
                "optimal_spot": round(S, 2),
                "pnl": pnl,
            }
        )

    return results


def score_roll_candidate(current_delta: float, candidate: dict) -> float:
    """Score a roll candidate. Higher = better.

    Weighs: delta reduction, net credit received, days of additional coverage.
    """
    delta_improvement = current_delta - candidate["delta"]
    net_credit = candidate.get("net_credit", 0)
    dte = candidate.get("dte", 0)
    return delta_improvement * 100 + net_credit * 10 + dte * 0.1


def find_best_rolls(
    current_short_strike: float,
    current_short_expiry: str,
    current_short_dte: int,
    current_short_price: float,
    current_delta: float,
    roll_chains: dict,
    spot: float,
    long_strike: float,
    long_cost: float,
    min_roll_dte: int,
    price_mode: str,
) -> list[dict]:
    """Find top-3 roll candidates satisfying both criteria:

    (a) delta < current short delta
    (b) net credit >= NET_CREDIT_MIN (not a large debit)

    Ranks by: delta reduction, net credit, DTE.
    """
    candidates = []

    for expiry, quotes in roll_chains.items():
        dte = days_to_expiry(expiry)
        if dte < min_roll_dte:
            continue
        if expiry == current_short_expiry:
            continue

        for quote in quotes:
            price = get_option_price(quote, price_mode)
            if not price or price <= 0:
                continue

            strike = quote["strike"]
            iv = calc_iv(price, spot, strike, dte, "C")
            if not iv:
                iv = estimate_iv(spot, strike, max(dte, 1) / 365, "call")
            if not iv:
                continue

            delta = abs(calc_delta(spot, strike, dte, iv, "C"))
            prob = calc_assignment_prob(spot, strike, dte, iv, "C")

            # Criterion (a): must have lower delta than current short
            if delta >= current_delta:
                continue

            net_credit = price - current_short_price

            # Criterion (b): not too large a debit
            if net_credit < NET_CREDIT_MIN:
                continue

            profit_per_day = price / dte if dte > 0 else 0

            # P&L if assigned at new short strike
            spread_width = strike - long_strike
            pnl_if_assigned = (spread_width - long_cost + price) * 100

            candidates.append(
                {
                    "strike": strike,
                    "expiry": expiry,
                    "dte": dte,
                    "price": round(price, 2),
                    "delta": round(delta, 4),
                    "assignment_prob": round(prob * 100, 1),
                    "iv_pct": round(iv * 100, 1),
                    "net_credit": round(net_credit, 2),
                    "profit_per_day": round(profit_per_day, 4),
                    "pnl_if_assigned": round(pnl_if_assigned, 2),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                }
            )

    candidates.sort(
        key=lambda c: score_roll_candidate(current_delta, c),
        reverse=True,
    )
    return candidates[:3]


def build_comparison_table(
    current: dict,
    rolls: list[dict],
    long_pos: dict,
) -> dict:
    """Side-by-side comparison of current short and up to 3 roll candidates."""

    def _entry(pos: dict) -> dict:
        long_strike = long_pos["strike"]
        long_cost = long_pos.get("avg_cost", 0)
        short_strike = pos["strike"]
        total_premium = pos.get("total_premium") or pos.get("price", 0)

        pnl_if_assigned = None
        if long_strike and short_strike > long_strike:
            spread_width = short_strike - long_strike
            pnl_if_assigned = round((spread_width - long_cost + total_premium) * 100, 2)

        return {
            "strike": short_strike,
            "expiry": pos.get("expiry"),
            "dte": pos.get("dte"),
            "delta": pos.get("delta"),
            "assignment_prob": pos.get("assignment_prob"),
            "profit_per_day": pos.get("profit_per_day"),
            "pnl_if_assigned": pnl_if_assigned,
        }

    result = {"current": _entry(current)}
    for i, roll in enumerate(rolls, 1):
        result[f"roll_{i}"] = _entry(roll)
    return result


# ===========================================================================
# DATA FETCHING (IBKR — separated from analytics above)
# ===========================================================================


async def _fetch_single_option_quote(
    ib, symbol: str, strike: float, expiry: str, right: str
) -> dict | None:
    """Fetch bid/ask/last and model Greeks for a single option contract."""
    from ib_async import Option

    contract = Option(symbol, expiry, strike, right, "SMART")
    qualified = await fetch_with_timeout(ib.qualifyContractsAsync(contract), timeout=10, default=[])
    if not qualified or not qualified[0] or not getattr(qualified[0], "conId", None):
        return None

    tickers = await fetch_with_timeout(ib.reqTickersAsync(qualified[0]), timeout=10, default=[])
    if not tickers:
        return None

    t = tickers[0]
    bid = t.bid if t.bid and t.bid > 0 else None
    ask = t.ask if t.ask and t.ask > 0 else None
    last = t.last if t.last and t.last > 0 else None

    ib_delta = None
    ib_iv_pct = None
    if t.modelGreeks:
        if t.modelGreeks.delta:
            ib_delta = round(t.modelGreeks.delta, 4)
        if t.modelGreeks.impliedVol:
            ib_iv_pct = round(t.modelGreeks.impliedVol * 100, 2)

    return {
        "bid": round(bid, 2) if bid is not None else None,
        "ask": round(ask, 2) if ask is not None else None,
        "last": round(last, 2) if last is not None else None,
        "ib_delta": ib_delta,
        "ib_iv_pct": ib_iv_pct,
    }


async def _fetch_option_quotes_batch(
    ib, symbol: str, expiry: str, strikes: list[float], right: str
) -> list[dict]:
    """Fetch option quotes for multiple strikes at one expiry — reuses roll.py pattern."""
    import logging

    from ib_async import Option

    contracts = [Option(symbol, expiry, s, right, "SMART") for s in strikes]

    ib_logger = logging.getLogger("ib_async")
    prev_level = ib_logger.level
    ib_logger.setLevel(logging.CRITICAL)

    try:
        qualified = await fetch_with_timeout(
            ib.qualifyContractsAsync(*contracts), timeout=15, default=[]
        )
    finally:
        ib_logger.setLevel(prev_level)

    qualified = [c for c in (qualified or []) if c is not None and getattr(c, "conId", None)]
    if not qualified:
        return []

    tickers = await fetch_with_timeout(ib.reqTickersAsync(*qualified), timeout=20, default=[])
    await asyncio.sleep(0.5)

    results = []
    for t in tickers or []:
        if not t.contract:
            continue
        bid = t.bid if t.bid and t.bid > 0 else None
        ask = t.ask if t.ask and t.ask > 0 else None
        mid = (bid + ask) / 2 if bid and ask else (bid or ask or 0)
        last = t.last if t.last and t.last > 0 else None
        results.append(
            {
                "strike": t.contract.strike,
                "expiry": expiry,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last": last,
            }
        )
    return sorted(results, key=lambda x: x["strike"])


def _identify_pmcc_spreads(normalized: list[dict]) -> list[dict]:
    """Identify PMCC (diagonal call) spreads: long LEAPS + short near-term call, same qty."""
    by_symbol = defaultdict(list)
    for pos in normalized:
        if pos["sec_type"] == "OPT" and pos.get("right") == "C":
            by_symbol[pos["symbol"]].append(pos)

    spreads = []
    for symbol, positions in by_symbol.items():
        longs = sorted(
            [p for p in positions if p["quantity"] > 0],
            key=lambda x: x.get("expiry", ""),
            reverse=True,
        )
        shorts = sorted(
            [p for p in positions if p["quantity"] < 0],
            key=lambda x: x.get("expiry", ""),
        )

        used = set()
        for long_pos in longs:
            for j, short_pos in enumerate(shorts):
                if j in used:
                    continue
                if (
                    long_pos["expiry"] > short_pos["expiry"]
                    and abs(long_pos["quantity"]) == abs(short_pos["quantity"])
                    and long_pos["strike"] < short_pos["strike"]
                ):
                    spreads.append(
                        {
                            "symbol": symbol,
                            "long": long_pos,
                            "short": short_pos,
                            "qty": int(abs(long_pos["quantity"])),
                        }
                    )
                    used.add(j)
                    break

    return spreads


async def _get_chain_params(ib, symbol: str) -> dict:
    """Fetch option chain expirations and strikes from IBKR."""
    from ib_async import Stock

    stock = Stock(symbol, "SMART", "USD")
    await fetch_with_timeout(ib.qualifyContractsAsync(stock), timeout=10, default=[])
    chains = await fetch_with_timeout(
        ib.reqSecDefOptParamsAsync(symbol, "", "STK", stock.conId), timeout=15, default=[]
    )
    if not chains:
        return {"expirations": [], "strikes": []}
    chain = best_option_chain(chains)
    return {
        "expirations": sorted(chain.expirations),
        "strikes": sorted(chain.strikes),
    }


async def get_pmcc_data(
    port: int = 7496,
    account: str | None = None,
    min_roll_dte: int = 7,
    price_mode: str = "mid",
) -> dict:
    """Fetch all PMCC positions and market data from IBKR, then run analytics.

    Returns structured JSON with per-spread risk analysis, daily P&L tables,
    roll candidates, and comparison tables.
    """
    try:
        async with ib_connection(port, CLIENT_IDS["pmcc_advisor"]) as ib:
            ib.reqMarketDataType(4)  # delayed-frozen fallback outside market hours
            await asyncio.sleep(2)

            managed = ib.managedAccounts()
            if account and account not in managed:
                return {
                    "generated_at": generated_at_str(),
                    "data_delay": "unknown",
                    "error": f"Account {account} not found. Available: {managed}",
                }
            accounts = [account] if account else list(managed)

            raw = await fetch_positions(ib, account=account)
            normalized = normalize_positions(raw)

            spreads = _identify_pmcc_spreads(normalized)
            if not spreads:
                return {
                    "generated_at": generated_at_str(),
                    "data_delay": "real-time",
                    "accounts": accounts,
                    "spreads": [],
                    "message": "No PMCC (diagonal call spread) positions found",
                }

            symbols = list({s["symbol"] for s in spreads})
            spot_prices = await fetch_spot_prices(ib, symbols)

            data_delay = "real-time"
            results = []

            for spread in spreads:
                symbol = spread["symbol"]
                long_pos = spread["long"]
                short_pos = spread["short"]
                qty = spread["qty"]
                spot = spot_prices.get(symbol)
                if not spot:
                    continue

                long_dte = days_to_expiry(long_pos["expiry"])
                short_dte = days_to_expiry(short_pos["expiry"])

                # --- Fetch quotes for both legs ---
                short_quote = await _fetch_single_option_quote(
                    ib, symbol, short_pos["strike"], short_pos["expiry"], "C"
                )
                long_quote = await _fetch_single_option_quote(
                    ib, symbol, long_pos["strike"], long_pos["expiry"], "C"
                )

                short_price = get_option_price(short_quote or {}, price_mode)
                long_price = get_option_price(long_quote or {}, price_mode)

                # --- Short leg analytics ---
                short_iv = None
                if short_price:
                    short_iv = calc_iv(short_price, spot, short_pos["strike"], short_dte, "C")
                if not short_iv and short_quote and short_quote.get("ib_iv_pct"):
                    short_iv = short_quote["ib_iv_pct"] / 100
                if not short_iv:
                    short_iv = estimate_iv(
                        spot, short_pos["strike"], max(short_dte, 1) / 365, "call"
                    )
                    data_delay = "stalled - using estimated IV"

                short_delta = calc_delta(spot, short_pos["strike"], short_dte, short_iv, "C")
                short_assign_prob = calc_assignment_prob(
                    spot, short_pos["strike"], short_dte, short_iv, "C"
                )

                # --- Long leg analytics ---
                long_iv = None
                if long_price:
                    long_iv = calc_iv(long_price, spot, long_pos["strike"], long_dte, "C")
                if not long_iv and long_quote and long_quote.get("ib_iv_pct"):
                    long_iv = long_quote["ib_iv_pct"] / 100
                if not long_iv:
                    long_iv = estimate_iv(spot, long_pos["strike"], max(long_dte, 1) / 365, "call")

                # --- Daily P&L table ---
                daily_pnl = []
                if short_dte >= 0:
                    daily_pnl = calc_daily_pnl_table(
                        long_strike=long_pos["strike"],
                        long_dte=long_dte,
                        long_cost=long_pos["avg_cost"],
                        long_iv=long_iv,
                        short_strike=short_pos["strike"],
                        short_dte=short_dte,
                        short_premium=abs(short_pos["avg_cost"]),
                        short_iv=short_iv,
                        qty=qty,
                        right="C",
                    )

                # --- Roll candidates ---
                chain_params = await _get_chain_params(ib, symbol)
                roll_exps = [
                    e
                    for e in chain_params.get("expirations", [])
                    if days_to_expiry(e) >= min_roll_dte
                    and e != short_pos["expiry"]
                    and e <= long_pos["expiry"]
                ][:5]

                all_strikes = chain_params.get("strikes", [])
                roll_strikes = [s for s in all_strikes if spot * 0.95 <= s <= spot * 1.25]

                roll_chains: dict[str, list] = {}
                for exp in roll_exps:
                    quotes = await _fetch_option_quotes_batch(ib, symbol, exp, roll_strikes, "C")
                    if quotes:
                        roll_chains[exp] = quotes

                rolls = []
                if short_price is not None:
                    rolls = find_best_rolls(
                        current_short_strike=short_pos["strike"],
                        current_short_expiry=short_pos["expiry"],
                        current_short_dte=short_dte,
                        current_short_price=short_price,
                        current_delta=abs(short_delta),
                        roll_chains=roll_chains,
                        spot=spot,
                        long_strike=long_pos["strike"],
                        long_cost=long_pos["avg_cost"],
                        min_roll_dte=min_roll_dte,
                        price_mode=price_mode,
                    )

                # --- Comparison table ---
                current_short_summary = {
                    "strike": short_pos["strike"],
                    "expiry": short_pos["expiry"],
                    "dte": short_dte,
                    "delta": round(abs(short_delta), 4),
                    "assignment_prob": round(short_assign_prob * 100, 1),
                    "price": round(short_price, 2) if short_price else None,
                    "profit_per_day": round(abs(short_pos["avg_cost"]) / max(short_dte, 1), 4),
                    "total_premium": abs(short_pos["avg_cost"]),
                }
                comparison = build_comparison_table(
                    current=current_short_summary,
                    rolls=rolls,
                    long_pos=long_pos,
                )

                results.append(
                    {
                        "symbol": symbol,
                        "account": account or accounts[0],
                        "qty": qty,
                        "underlying_price": round(spot, 2),
                        "long": {
                            "strike": long_pos["strike"],
                            "expiry": long_pos["expiry"],
                            "dte": long_dte,
                            "avg_cost": long_pos["avg_cost"],
                            "current_price": round(long_price, 2) if long_price else None,
                            "iv_pct": round(long_iv * 100, 1) if long_iv else None,
                            "ib_delta": long_quote.get("ib_delta") if long_quote else None,
                            "ib_iv_pct": long_quote.get("ib_iv_pct") if long_quote else None,
                        },
                        "short": {
                            "strike": short_pos["strike"],
                            "expiry": short_pos["expiry"],
                            "dte": short_dte,
                            "premium_received": abs(short_pos["avg_cost"]),
                            "current_price": round(short_price, 2) if short_price else None,
                            "iv_pct": round(short_iv * 100, 1) if short_iv else None,
                            "delta": round(abs(short_delta), 4),
                            "assignment_prob_pct": round(short_assign_prob * 100, 1),
                            "ib_delta": short_quote.get("ib_delta") if short_quote else None,
                            "ib_iv_pct": short_quote.get("ib_iv_pct") if short_quote else None,
                        },
                        "daily_pnl": daily_pnl,
                        "roll_candidates": rolls,
                        "comparison": comparison,
                    }
                )

            return {
                "generated_at": generated_at_str(),
                "data_delay": data_delay,
                "accounts": accounts,
                "price_mode": price_mode,
                "min_roll_dte": min_roll_dte,
                "spreads": results,
            }

    except ConnectionError as e:
        return {
            "generated_at": generated_at_str(),
            "data_delay": "unknown",
            "error": f"{e}. Is TWS/Gateway running?",
        }
