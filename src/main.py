import ingestion, preprocessing, processing, utils, model
import os
import sys
import shutil  # Added for directory removal
from discord_bot import run_discord_bot
from config import load_config, validate_args

def process_stock(ticker="AAPL", period="5y", interval="1d"):
    """Process a single stock"""
    # Download data
    print(f"Processing {ticker} with period {period} and interval {interval}")
    data = ingestion.download_stock_data(ticker, period, interval)
    
    # Clean data
    preprocessing.cleanData(data, ticker, period, interval)
    
    # Process data and return results
    # Add your processing logic here
    return {"status": "success", "ticker": ticker}

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    mode = config.get('mode', 'standalone')
    stock_config = config.get('default_stock', {})
    
    # Reset data directories
    if os.path.exists('./data/raw'):
        shutil.rmtree('./data/raw')
    if os.path.exists('./data/processed'):
        shutil.rmtree('./data/processed')
        
    # Create fresh data directories
    os.makedirs('./data/raw', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    
    if mode == 'discord':
        # Get token from config with fallback to environment
        token = config.get('discord', {}).get('token') or os.getenv("DISCORD_TOKEN")
            
        if not token:
            print("Please provide a Discord token in config.json or DISCORD_TOKEN environment variable")
        else:
            print("Starting Discord bot...")
            run_discord_bot(token)
    else:
        # Process single stock using config settings
        ticker = stock_config.get('ticker', "AAPL")
        period = stock_config.get('period', "5y")
        interval = stock_config.get('interval', "1d")
        
        # Validate arguments
        valid, error_msg = validate_args(period, interval, config)
        if not valid:
            print(f"Error: {error_msg}")
            sys.exit(1)
        
        print(f"Processing stock {ticker} for {period} with {interval} interval...")
        process_stock(ticker, period, interval)

