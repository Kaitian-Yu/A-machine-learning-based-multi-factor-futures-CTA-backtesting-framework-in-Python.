
def backtest_futures_static_threshold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    datetime_col="time",
    symbol_col="symbol",
    open_col="open",
    high_col="high",
    low_col="low",
    close_col="close",
    score_col="factor_ml",
    initial_capital: float = 1_000_000.0,
    leverage: float = 3.0,
    max_pos: int = 8,
    base_w: float = 0.02,
    cap_w: float = 0.05,
    th_percentile: float = 90.0,
    rank_drop_exit: float = 0.10,
    fee_rate: float = 0.0005,
    slip_rate: float = 0.000,
):
    """
    v3 执行逻辑：
    - BUY = If the current bar closes, a limit order will be executed on the next bar.
      However, the following condition must be met: next_low <= trigger_price <= next_high; otherwise, the limit order will not be executed.
    - SELL = Next bar high~low random transaction price × (1 - slip_rate)
    """

    # ====================== Utility functions======================
    def _prep(raw: pd.DataFrame, tag: str):
        df = raw.copy()
        df = df.rename(columns={
            datetime_col: "datetime",
            symbol_col: "symbol",
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            score_col: "score",
        })
        df["dataset"] = tag
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", "symbol", "close", "score"])
        df = (
            df.sort_values(["datetime", "symbol"])
              .drop_duplicates(subset=["datetime", "symbol"], keep="last")
              .sort_values(["symbol", "datetime"])
              .reset_index(drop=True)
        )
        return df

    train = _prep(train_df, "train")
    test  = _prep(test_df,  "test")

    # ======================Threshold calculation ======================
    stats = (
        train.groupby("symbol")["score"]
             .agg(th=lambda s: np.nanpercentile(s.to_numpy(), th_percentile),
                  mx="max")
             .reset_index()
    )

    stats["mx"] = np.where(stats["mx"] <= stats["th"], stats["th"] + 1e-9, stats["mx"])

    test = test.merge(stats, on="symbol", how="left")
    test = test.dropna(subset=["th", "mx"]).reset_index(drop=True)

    all_times = np.sort(test["datetime"].unique())

    # ====================== Initialize account======================
    equity = float(initial_capital)
    positions = {}
    equity_records = []
    trade_records = []
    pos_snapshots = []
    missed_buys = []

    # ====================== Main loop======================
    for i, t in enumerate(all_times):

        snap = (
            test.loc[test["datetime"] == t]
                .sort_values("symbol")
                .drop_duplicates(subset="symbol", keep="last")
        )

        # next bar
        if i < len(all_times) - 1:
            next_t = all_times[i + 1]
            next_snap = (
                test.loc[test["datetime"] == next_t]
                    .sort_values("symbol")
                    .drop_duplicates(subset="symbol", keep="last")
            )
        else:
            next_snap = None

        # --------- 1)Profit and loss per bar of holding positions ----------
        total_pnl_bar = 0.0
        for sym, pos in list(positions.items()):
            row = snap.loc[snap["symbol"] == sym]
            if row.empty:
                continue
            price_now = float(row["close"].iloc[0])
            delta_pnl = pos["qty"] * (price_now - pos["last_price"])
            total_pnl_bar += delta_pnl
            pos["last_price"] = price_now

        equity += total_pnl_bar

        # --------- 2) Cross-section ranking ----------
        snap["cs_rank"] = snap["score"].rank(pct=True, method="first")

        # --------- 3) Closing a position: Selling = Market order + Slippage----------
        for sym, pos in list(positions.items()):

            row_curr = snap.loc[snap["symbol"] == sym]
            if row_curr.empty:
                continue

            curr_rank = float(row_curr["cs_rank"].iloc[0])

            # Triggering a liquidation signal
            if curr_rank <= pos["entry_rank"] - rank_drop_exit:

                if next_snap is not None:
                    row_next = next_snap.loc[next_snap["symbol"] == sym]
                else:
                    row_next = pd.DataFrame()

                if next_snap is None or row_next.empty:
                    raw_base_price = float(row_curr["close"].iloc[0])
                else:
                    next_high = float(row_next["high"].iloc[0])
                    next_low  = float(row_next["low"].iloc[0])
                    raw_base_price = random.uniform(next_low, next_high)

                sell_price = raw_base_price * (1 - slip_rate)

                notional_exit = abs(pos["qty"]) * sell_price
                exit_fee = notional_exit * fee_rate
                gross_pnl = pos["qty"] * (sell_price - pos["entry_price"])
                net_pnl = gross_pnl - pos["entry_fee"] - exit_fee

                equity -= exit_fee

                trade_records.append({
                    "datetime": t,
                    "symbol": sym,
                    "side": "SELL",
                    "price": sell_price,
                    "qty": pos["qty"],
                    "notional": notional_exit,
                    "fee": exit_fee,
                    "pnl": net_pnl
                })

                positions.pop(sym, None)

        # --------- 4) Account risk value ----------
        exposure = sum(abs(pos["qty"]) * pos["last_price"] for pos in positions.values())
        margin_required = exposure / leverage if leverage > 0 else 0.0
        capital_pool = equity - margin_required
        exposure_limit = equity * leverage

        # --------- 5) BUY: Next bar open + limit order logic ----------
        cand = snap[snap["score"] >= snap["th"]].copy()
        if positions:
            cand = cand[~cand["symbol"].isin(positions)]
        cand = cand.sort_values("score", ascending=False)

        remaining_cap = max(0.0, exposure_limit - exposure)

        for _, r in cand.iterrows():
            if len(positions) >= max_pos:
                break
            if remaining_cap <= 0:
                break
            if next_snap is None:
                break

            row_next_buy = next_snap.loc[next_snap["symbol"] == r["symbol"]]
            if row_next_buy.empty:
                missed_buys.append({"datetime": t, "symbol": r["symbol"], "reason": "no_next_bar"})
                continue

            next_open = float(row_next_buy["open"].iloc[0])
            next_low  = float(row_next_buy["low"].iloc[0])
            next_high = float(row_next_buy["high"].iloc[0])

            trigger_price = float(r["close"])

            # === The limit order must be triggered. ===
            if not (next_low <= trigger_price <= next_high):
                missed_buys.append({
                    "datetime": t,
                    "symbol": r["symbol"],
                    "trigger_price": trigger_price,
                    "next_low": next_low,
                    "next_high": next_high,
                    "reason": "limit_not_touched"
                })
                continue

            # === make a deal ===
            th = float(r["th"])
            mx = float(r["mx"])
            score = float(r["score"])

            ramp = (score - th) / (mx - th)
            ramp = float(np.clip(ramp, 0.0, 1.0))

            weight = base_w + (cap_w - base_w) * ramp
            weight = float(np.clip(weight, base_w, cap_w))
            #Add rolling rate target
            weight = weight * r['vol_scale']
            #add risk Parity
            weight = weight * r['rp_weight']
            
            target_notional = weight * equity * leverage
            notional = min(target_notional, remaining_cap)
            if notional <= 0:
                continue

            entry_price = next_open
            qty = notional / entry_price
            entry_fee = notional * fee_rate

            equity -= entry_fee

            positions[r["symbol"]] = {
                "entry_time": t,
                "entry_price": entry_price,
                "entry_score": score,
                "entry_rank": float(r["cs_rank"]),
                "qty": qty,
                "last_price": entry_price,
                "entry_fee": entry_fee,
                "th": th,
                "mx": mx
            }

            trade_records.append({
                "datetime": t,
                "symbol": r["symbol"],
                "side": "BUY",
                "price": entry_price,
                "qty": qty,
                "notional": notional,
                "fee": entry_fee,
                "pnl": 0.0
            })

            remaining_cap -= notional

        # --------- 6)Record account information ----------
        equity_records.append({
            "datetime": t,
            "capital_pool": capital_pool,
            "positions_value": exposure,
            "equity": equity,
            "num_positions": len(positions),
            "exposure": exposure,
            "exposure_limit": exposure_limit
        })

        # --------- 7) Portfolio snapshot ----------
        for sym, pos in positions.items():
            row = snap.loc[snap["symbol"] == sym]
            if row.empty:
                curr_score = np.nan
                curr_rank = np.nan
            else:
                curr_score = float(row["score"].iloc[0])
                curr_rank = float(row["cs_rank"].iloc[0])

            pos_snapshots.append({
                "datetime": t,
                "symbol": sym,
                "entry_time": pos["entry_time"],
                "entry_score": pos["entry_score"],
                "entry_rank": pos["entry_rank"],
                "current_score": curr_score,
                "current_rank": curr_rank,
                "qty": pos["qty"],
                "last_price": pos["last_price"],
                "notional": abs(pos["qty"]) * pos["last_price"],
                "th": pos["th"],
                "mx": pos["mx"]
            })

    # --------- Final output ----------
    equity_df = pd.DataFrame(equity_records).set_index("datetime")
    trades_df = pd.DataFrame(trade_records)
    pos_log = pd.DataFrame(pos_snapshots)
    missed_buys_df = pd.DataFrame(missed_buys)

    return equity_df, trades_df, pos_log, missed_buys_df
