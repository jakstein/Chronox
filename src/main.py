import os
import sys
import shutil 
from discord_bot import runDiscordBot
from config import loadConfig

if __name__ == "__main__":
    # wczytaj konfigurację
    config = loadConfig()
    
    # resetuj katalogi danych
    if os.path.exists('./data/raw'):
        shutil.rmtree('./data/raw')
    if os.path.exists('./data/processed'):
        shutil.rmtree('./data/processed')
    if os.path.exists('./data/predictions'):
        shutil.rmtree('./data/predictions')    
        # utwórz nowe katalogi danych
    os.makedirs('./data/raw', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    os.makedirs('./data/predictions', exist_ok=True)
    
    # pobierz token z konfiguracji z rezerwą z zmiennych środowiskowych
    token = config.get('discord', {}).get('token') or os.getenv("DISCORD_TOKEN")
        
    if not token:
        print("Proszę podać token Discorda w pliku config.json lub w zmiennej środowiskowej DISCORD_TOKEN")
        sys.exit(1)
    else:
        print("Uruchamianie bota Discord...")
        runDiscordBot(token)

