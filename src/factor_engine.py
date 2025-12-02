#adding volatility
def apply_vol_target(df, target_vol=0.12, window=60):
    df = df.copy()

    df["ret"] = df["close"].pct_change()

    ann_factor = np.sqrt(16000)  

    df["real_vol"] = (
        df["ret"].rolling(window)
        .std()
        * ann_factor
    )


    df["vol_scale"] = target_vol / df["real_vol"]
    df["vol_scale"] = df["vol_scale"].clip(upper=5, lower=0)  

    return df["vol_scale"]


def add_tech_factors(df, short_win=12, long_win=48, vol_win=20):
    """
    df: include [symbol, time, open, high, low, close, vol]  DataFrame
   `short_win` and `long_win` are MA windows; 5 min * 12 ≈ 1 hour, 5 min * 48 ≈ 4 hours, these are just examples.
    """

    df = df.sort_values(['symbol', 'time']).copy()
    g = df.groupby('symbol', group_keys=False)

    # ---- 2.1 Benefit factors ----
    df['ret_1'] = g['close'].pct_change(1)         

    df['MTM_long'] = df['close'] - df['close'].shift(15)
    df["vol_scale"] = apply_vol_target(df, target_vol=0.11, window=60)

    return df


