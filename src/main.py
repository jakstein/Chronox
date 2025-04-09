import ingestion, preprocessing, processing, utils, model
import os
import sys
import shutil 
from discord_bot import runDiscordBot
from config import loadConfig, validateArgs

def process_stock(ticker="AAPL", period="5y", interval="1d"):
    # dl data
    print(f"Processing {ticker} with period {period} and interval {interval}")
    data = ingestion.download_stock_data(ticker, period, interval)
    
    # clean data
    preprocessing.cleanData(data, ticker, period, interval)
    
    return {"status": "success", "ticker": ticker}

if __name__ == "__main__":
    # load configuration
    config = loadConfig()
    mode = config.get('mode', 'standalone')
    stock_config = config.get('default_stock', {})
    
    # reset data directories
    if os.path.exists('./data/raw'):
        shutil.rmtree('./data/raw')
    if os.path.exists('./data/processed'):
        shutil.rmtree('./data/processed')
    if os.path.exists('./data/predictions'):
        shutil.rmtree('./data/predictions')

    # create fresh data directories
    os.makedirs('./data/raw', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    os.makedirs('./data/predictions', exist_ok=True)
    
    if mode == 'discord':
        # get token from config with fallback to environment
        token = config.get('discord', {}).get('token') or os.getenv("DISCORD_TOKEN")
            
        if not token:
            print("Please provide a Discord token in config.json or DISCORD_TOKEN environment variable")
        else:
            print("Starting Discord bot...")
            runDiscordBot(token)
    else:
        # process single stock using config settings
        ticker = stock_config.get('ticker', "AAPL")
        period = stock_config.get('period', "5y")
        interval = stock_config.get('interval', "1d")
        
        # validate arguments
        valid, error_msg = validateArgs(period, interval, config)
        if not valid:
            print(f"Error: {error_msg}")
            sys.exit(1)
        
        print(f"Processing stock {ticker} for {period} with {interval} interval...")
        process_stock(ticker, period, interval)

