import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logger imports
from utils.logger import Logger
logger = Logger.get_logger()

# Internal imports
from data.data_manager import DataManager
from visualization.data_visualization import DataVisualization
from config.tickers import DOW_30_TICKER, NASDAQ_100_TICKER, SP_500_TICKER
from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    PROCESSOR_PARAMS,
    NORMALIZATION_PARAMS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize financial data distributions and properties")
    parser.add_argument("--assets", type=str, nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                       help="Ticker symbols to analyze")
    parser.add_argument("--start-date", type=str, default=TRAIN_START_DATE,
                       help="Start date for data")
    parser.add_argument("--end-date", type=str, default=TEST_END_DATE,
                       help="End date for data")
    parser.add_argument("--processors", type=str, nargs="+", default=["technical_indicator", "vix"],
                       help="Data processors to apply")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Whether to normalize data")
    parser.add_argument("--normalize-method", type=str, default=NORMALIZATION_PARAMS["method"],
                       help="Normalization method (zscore, rolling, percentage)")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--selected-columns", type=str, nargs="+", default=None,
                       help="Specific columns to visualize (defaults to all)")
    parser.add_argument("--time-series-columns", type=str, nargs="+", default=["close", "volume"],
                       help="Columns to use for time series visualization")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    logger.info(f"Starting data visualization for assets: {args.assets}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Initialize data manager
    data_manager = DataManager(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        normalized_data_dir="data/normalized",
        save_raw_data=True,
        save_processed_data=True,
        save_normalized_data=True,
        use_saved_data=True,
    )
    
    # Download raw data
    logger.info("Downloading raw data...")
    raw_data = data_manager.download_data(
        tickers=args.assets,
        start_date=args.start_date,
        end_date=args.end_date,
        source="yahoo",
        time_interval="1d",
        add_day_index_bool=True,
    )
    
    # Process with technical indicators
    logger.info("Processing data with technical indicators...")
    processed_data = data_manager.process_data(
        data=raw_data,
        processors=args.processors,
        processor_params=PROCESSOR_PARAMS,
        save_data=True,
    )
    
    # Normalize data if requested
    normalized_data = None
    if args.normalize:
        logger.info(f"Normalizing data with method: {args.normalize_method}")
        normalized_data = data_manager.normalize_data(
            data=processed_data, 
            method=args.normalize_method,
            save_data=True,
            handle_outliers=True,
            fill_value=0
        )
    
    # Initialize visualization tool
    logger.info(f"Initializing visualization tool with output directory: {args.output_dir}")
    visualizer = DataVisualization(save_dir=args.output_dir)
    
    # Generate timestamp for consistent filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"analysis_{timestamp}"
    
    # Display basic information about the datasets
    logger.info("Dataset information:")
    logger.info(f"Raw data shape: {raw_data.shape}")
    logger.info(f"Processed data shape: {processed_data.shape}")
    if normalized_data is not None:
        logger.info(f"Normalized data shape: {normalized_data.shape}")
    
    # Get unique tickers in the data
    tickers = raw_data['ticker'].unique().tolist()
    
    # Run all visualizations
    logger.info("Generating visualizations...")
    visualizer.visualize_all(
        raw_data=raw_data,
        processed_data=processed_data,
        normalized_data=normalized_data,
        columns=args.selected_columns,
        time_series_columns=args.time_series_columns,
        tickers=tickers,
        output_prefix=output_prefix
    )
    
    logger.info(f"All visualizations completed and saved to {args.output_dir}")


if __name__ == "__main__":
    main() 