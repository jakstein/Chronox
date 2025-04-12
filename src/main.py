import os
import sys
import shutil 
from discord_bot import runDiscordBot
from config import loadConfig

if __name__ == "__main__":
    # load configuration
    config = loadConfig()
    
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
    
    # get token from config with fallback to environment
    token = config.get('discord', {}).get('token') or os.getenv("CHRONOX_DISCORD_TOKEN")
        
    if not token:
        print("Please provide a Discord token in config.json or CHRONOX_DISCORD_TOKEN environment variable")
        sys.exit(1)
    else:
        print("Starting Discord bot...")
        runDiscordBot(token)

