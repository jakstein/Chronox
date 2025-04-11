import json
import os

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Nie znaleziono pliku konfiguracyjnego w {config_path}. Używam domyślnej konfiguracji.")
    except json.JSONDecodeError:
        print(f"Błąd parsowania pliku konfiguracyjnego w {config_path}. Używam domyślnej konfiguracji.")
      # domyślna konfiguracja jako awaryjne rozwiązanie
    return {
        "mode": "discord",
        "discord": {
            "token": os.getenv("DISCORD_TOKEN", ""),
            "command_prefix": "!"
        },
        "default_stock": {
            "ticker": "AAPL",
            "period": "1y",
            "interval": "1d"
        },
        "news_sentiment": {
            "enabled": True,
            "storiesCount": 20,
            "daysLookback": 7,
            "sentiment_impact_weight": 0.5
        },
        "allowed_arguments": {
            "intervals": ["1d", "5d", "1wk", "1mo"],
            "periods": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
            "long_periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            "short_intervals": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        }
    }

def validateArgs(period, interval, config=None):
    if config is None:
        config = loadConfig()
        
    allowed_args = config.get('allowed_arguments', {})
    allowed_intervals = allowed_args.get('intervals', [])
    allowed_periods = allowed_args.get('periods', [])
    long_periods = allowed_args.get('long_periods', [])
    short_intervals = allowed_args.get('short_intervals', [])

    if period and period not in allowed_periods:
        return False, f"Nieprawidłowy okres: '{period}'. Dozwolone okresy to: {', '.join(allowed_periods)}"
    
    if interval and interval not in allowed_intervals:
        return False, f"Nieprawidłowy interwał: '{interval}'. Dozwolone interwały to: {', '.join(allowed_intervals)}"
    
    # sprawdzenie niekompatybilnych kombinacji okres-interwał
    if period in long_periods and interval in short_intervals:
        return False, f"Krótkie interwały ({', '.join(short_intervals)}) nie są dozwolone z długimi okresami ({', '.join(long_periods)}). Proszę wybrać dłuższy interwał (np. 1d, 5d, 1wk, 1mo)."
    
    return True, None
