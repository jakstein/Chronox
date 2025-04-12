import os
import sys
import shutil 
from discord_bot import runDiscordBot
from config import loadConfig

# UTF-8 dla znaków polskich
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

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
    
    # pobierz token z pliku konfiguracyjnego z fallbackiem na zmiennych środowiskowych
    token = config.get('discord', {}).get('token') or os.getenv("CHRONOX_DISCORD_TOKEN")
        
    if not token:
        print("Proszę podać token Discorda w pliku config.json lub w zmiennej środowiskowej CHRONOX_DISCORD_TOKEN")
        sys.exit(1)
    else:
        print("Uruchamianie bota Discord...")
        runDiscordBot(token)

