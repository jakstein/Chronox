import discord
from discord.ext import commands
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import ingestion, preprocessing, processing, utils, model
from config import load_config, validate_args

# configure Discord bot based on config
config = load_config()
discord_config = config.get('discord', {})

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=discord_config.get('command_prefix', '!'), intents=intents)

# Load default values from config
stock_config = config.get('default_stock', {})
default_ticker = stock_config.get('ticker', 'AAPL')
default_period = stock_config.get('period', '5y')
default_interval = stock_config.get('interval', '1d')

# Load allowed arguments from config
allowed_args = config.get('allowed_arguments', {})
allowed_intervals = allowed_args.get('intervals', [])
allowed_periods = allowed_args.get('periods', [])

@bot.event
async def on_ready():
    print(f'Bot is ready. Logged in as {bot.user}')

@bot.command(name='help_stock')
async def help_stock(ctx):
    """Show help for all stock commands"""
    help_text = """
**Chronox Stock Bot Commands**
`!fetch_stock <ticker> [period] [interval]` - Fetch stock data
   - ticker: Stock symbol (e.g., AAPL)
   - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
   - interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo (default: 1d)

`!stock_chart <ticker> [period] [interval]` - Get stock price chart

`!stock_features <ticker> [period] [interval]` - Get feature table for stock

`!predict_xgboost <ticker> [period] [interval] [days_ahead] [test_size]` - Get XGBoost prediction
   - days_ahead: Days to predict ahead (default: 30)
   - test_size: Test size ratio (default: 0.2)

`!predict_lightgbm <ticker> [period] [interval] [days_ahead] [test_size]` - Get LightGBM prediction
    """
    await ctx.send(help_text)

@bot.command(name='fetch_stock')
async def fetch_stock_cmd(ctx, ticker=None, period=None, interval=None):
    """Fetch stock data for the given ticker"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # Validate arguments
    valid, error_msg = validate_args(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    await ctx.send(f"Fetching {ticker} stock data for {period} with {interval} interval...")
    
    try:
        os.makedirs('./data/raw', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        
        ingestion.fetchStock(ticker, period, interval)
        await ctx.send(f"Successfully fetched {ticker} data!")
        
        file = discord.File(f'./data/raw/{ticker}_{period}_{interval}.png', filename=f'{ticker}_chart.png')
        await ctx.send(file=file)
    except Exception as e:
        await ctx.send(f"Error fetching stock data: {str(e)}")

@bot.command(name='stock_chart')
async def stock_chart(ctx, ticker=None, period=None, interval=None):
    """Send stock chart for the given ticker"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # Validate arguments
    valid, error_msg = validate_args(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        # Always fetch fresh data
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        chart_path = f'./data/raw/{ticker}_{period}_{interval}.png'
        file = discord.File(chart_path, filename=f'{ticker}_chart.png')
        await ctx.send(file=file)
    except Exception as e:
        await ctx.send(f"Error generating chart: {str(e)}")

@bot.command(name='stock_features')
async def stock_features(ctx, ticker=None, period=None, interval=None):
    """Display feature table for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # Validate arguments
    valid, error_msg = validate_args(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        # Always fetch fresh data
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        # Process the data
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Calculating features...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
    
        data = pd.read_csv(processed_path)
        
        # last 10 rows for display
        last_rows = data.tail(10)
        
        # format as a text table for Discord
        features_str = f"**{ticker} Features (last 10 rows)**\n```\n"
        features_str += last_rows.to_string(index=False)
        features_str += "\n```"
        
        if len(features_str) > 2000:
            # in case too long for Discord, send as a CSV file
            buffer = io.StringIO()
            last_rows.to_csv(buffer, index=False)
            buffer.seek(0)
            
            file = discord.File(
                fp=io.BytesIO(buffer.getvalue().encode()),
                filename=f"{ticker}_features.csv"
            )
            await ctx.send("Features table (last 10 rows):", file=file)
        else:
            await ctx.send(features_str)
            
    except Exception as e:
        await ctx.send(f"Error generating features: {str(e)}")

@bot.command(name='predict_xgboost')
async def predict_xgboost(ctx, ticker=None, period=None, interval=None, days_ahead=30, test_size=0.2):
    """Get XGBoost prediction for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # Validate arguments
    valid, error_msg = validate_args(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        days_ahead = int(days_ahead)
        test_size = float(test_size)
        
        # Always fetch fresh data
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        # Process the data
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Processing data with features...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
        
        # Remove duplicate code - this was repeated in the original
        await ctx.send(f"Running XGBoost prediction for {ticker}, {days_ahead} days ahead...")
        result = model.trainXGBoost(data, ticker, period, interval, days_ahead, test_size, return_result=True)
        
        # format the prediction message
        prediction_msg = f"""
**XGBoost Prediction for {result['ticker']}**
- Period: {result['period']}, Interwał: {result['interval']}
- Last Close Price: ${result['last_close']:.2f}
- Predicted Price ({days_ahead} days ahead): ${result['prediction']:.2f}
- Change: ${(result['prediction'] - result['last_close']):.2f} ({((result['prediction'] / result['last_close'] - 1) * 100):.2f}%)
- Model Error (RMSE): {result['error']:.4f}
 """
        await ctx.send(prediction_msg)
        
    except Exception as e:
        await ctx.send(f"Error making prediction: {str(e)}")

@bot.command(name='predict_lightgbm')
async def predict_lightgbm(ctx, ticker=None, period=None, interval=None, days_ahead=30, test_size=0.2):
    """Get LightGBM prediction for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # Validate arguments
    valid, error_msg = validate_args(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        days_ahead = int(days_ahead)
        test_size = float(test_size)
        
        # Always fetch fresh data
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        # Process the data
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Processing data with features...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
        
        # run prediction
        await ctx.send(f"Running LightGBM prediction for {ticker}, {days_ahead} days ahead...")
        result = model.trainLightGBM(data, ticker, period, interval, days_ahead, test_size, return_result=True)
        
        # format the prediction message
        prediction_msg = f"""
**LightGBM Prediction for {result['ticker']}**
- Period: {result['period']}, Interval: {result['interval']}
- Last Close Price: ${result['last_close']:.2f}
- Predicted Price ({days_ahead} days ahead): ${result['prediction']:.2f}
- Change: ${(result['prediction'] - result['last_close']):.2f} ({((result['prediction'] / result['last_close'] - 1) * 100):.2f}%)
- Model Error (RMSE): {result['error']:.4f}
"""
        await ctx.send(prediction_msg)
        
    except Exception as e:
        await ctx.send(f"Error making prediction: {str(e)}")

@bot.command(name='prune')
async def prune(ctx, count="5"):
    try:
        count = int(count)
        if count <= 0:
            await ctx.send("Please provide a positive number of messages to delete.")
            return
            
        # Get recent messages from the channel
        messages = []
        async for message in ctx.channel.history(limit=100):
            messages.append(message)
        
        # Filter for bot messages only
        bot_messages = [message for message in messages if message.author == bot.user]
        
        # Select the number to delete (up to the requested count)
        messages_to_delete = bot_messages[:min(count, len(bot_messages))]
        
        # Delete the messages
        deleted_count = 0
        for message in messages_to_delete:
            await message.delete()
            deleted_count += 1
            
        confirmation_msg = await ctx.send(f"Deleted {deleted_count} bot messages.")
        
        # Delete the command message too
        await ctx.message.delete()
        
    except Exception as e:
        await ctx.send(f"Error pruning messages: {str(e)}")

def run_discord_bot(token=None):
    """Run the Discord bot with the given token or from config"""
    if token is None:
        token = discord_config.get('token', os.getenv("DISCORD_TOKEN"))
    
    if not token:
        print("No Discord token found in config or environment variables")
        return
        
    bot.run(token)

if __name__ == "__main__":
    # Get token from config, with fallback to environment variable
    token = discord_config.get('token')
    if not token:
        token = os.getenv("DISCORD_TOKEN")
        
    if not token:
        print("Please set a Discord token in config.json or DISCORD_TOKEN environment variable")
    else:
        run_discord_bot(token)
