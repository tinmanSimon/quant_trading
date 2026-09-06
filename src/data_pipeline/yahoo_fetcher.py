from .base_fetcher import OHLCV_SCHEMA, BaseDataProvider
import yfinance as yf
import polars as pl
import pandas as pd

class YFinanceProvider(BaseDataProvider):
    def fetch_ohlcv(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = "1h"
    ) -> pl.DataFrame:
        # yfinance expects intervals like '1h', '1d'
        df_pandas = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval=timeframe,
            progress=False,
            auto_adjust=False  # Keep raw unadjusted values for execution modeling
        )
        
        if df_pandas.empty:
            raise ValueError(f"No data returned for {symbol} from yfinance.")

        # Flatten yfinance MultiIndex columns if present
        if isinstance(df_pandas.columns, pd.MultiIndex):
            df_pandas.columns = df_pandas.columns.get_level_values(0)

        df_pandas = df_pandas.reset_index()

        # Convert to Polars and map to canonical schema
        df = pl.from_pandas(df_pandas)
        
        # Identify the timestamp column name ('Date' or 'Datetime')
        time_col = "Datetime" if "Datetime" in df.columns else "Date"

        df = df.select([
            pl.col(time_col).dt.convert_time_zone("UTC").alias("timestamp"),
            pl.lit(symbol).alias("symbol"),
            pl.col("Open").cast(pl.Float64).alias("open"),
            pl.col("High").cast(pl.Float64).alias("high"),
            pl.col("Low").cast(pl.Float64).alias("low"),
            pl.col("Close").cast(pl.Float64).alias("close"),
            pl.col("Volume").cast(pl.Float64).alias("volume"),
        ])

        return df
