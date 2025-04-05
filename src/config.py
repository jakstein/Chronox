import json
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using default configuration.")
    except json.JSONDecodeError:
        print(f"Error parsing config file at {config_path}. Using default configuration.")
    
    # Default config as fallback
    return {
        "mode": "standalone",
        "discord": {
            "token": os.getenv("DISCORD_TOKEN", ""),
            "command_prefix": "!"
        },
        "default_stock": {
            "ticker": "AAPL",
            "period": "1y",
            "interval": "1d"
        },
        "allowed_arguments": {
            "intervals": ["1d", "5d", "1wk", "1mo"],
            "periods": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
            "long_periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            "short_intervals": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        }
    }

def validate_args(period, interval, config=None):
    """Validate if period and interval are in the allowed lists"""
    if config is None:
        config = load_config()
        
    allowed_args = config.get('allowed_arguments', {})
    allowed_intervals = allowed_args.get('intervals', [])
    allowed_periods = allowed_args.get('periods', [])
    long_periods = allowed_args.get('long_periods', [])
    short_intervals = allowed_args.get('short_intervals', [])

    if period and period not in allowed_periods:
        return False, f"Invalid period: '{period}'. Allowed periods are: {', '.join(allowed_periods)}"
    
    if interval and interval not in allowed_intervals:
        return False, f"Invalid interval: '{interval}'. Allowed intervals are: {', '.join(allowed_intervals)}"
    
    # Check for incompatible period-interval combinations
    if period in long_periods and interval in short_intervals:
        return False, f"Short intervals ({', '.join(short_intervals)}) are not allowed with long periods ({', '.join(long_periods)}). Please select a longer interval (e.g., 1d, 5d, 1wk, 1mo)."
    
    return True, None
