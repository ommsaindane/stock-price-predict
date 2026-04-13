from src.data_loader import StockDataConfig, load_stock_data
from src.feature_engineering import FeatureEngineeringConfig, engineer_features


def main() -> None:
    data_config = StockDataConfig()
    feature_config = FeatureEngineeringConfig()

    clean_data = load_stock_data(data_config)
    enriched_data = engineer_features(clean_data, feature_config)

    print(f"Enriched shape: {enriched_data.shape}")
    print(
        "Engineered columns: "
        f"MA{feature_config.ma_short_window}, "
        f"MA{feature_config.ma_long_window}, "
        f"EMA{feature_config.ema_span}"
    )
    print(f"NaNs remaining: {int(enriched_data.isna().sum().sum())}")


if __name__ == "__main__":
    main()
