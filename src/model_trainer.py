

def ml_mine_factors_simple(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.3,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
):
    df = df.sort_values(['time', 'symbol']).copy()

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    # Time series segmentation: 70% training, 30% testing
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # Machine Learning Factor: The model's prediction of future returns
    df['factor_ml'] = model.predict(X)

    # A simple check of the model's fit (not the strategy's return).
    y_pred_train = df['factor_ml'].iloc[:split_idx]
    y_pred_test = df['factor_ml'].iloc[split_idx:]

    print("\nML Model Performance (for checking for overfitting only) ====")
    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test  R2:", r2_score(y_test, y_pred_test))
    print("Train MSE:", mean_squared_error(y_train, y_pred_train))
    print("Test  MSE:", mean_squared_error(y_test, y_pred_test))

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n==== Top 20 Feature importance (candidate factors)====")
    print(importance_df.head(20))

    return {
        "model": model,
        "importance": importance_df,
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
