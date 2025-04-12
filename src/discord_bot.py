import discord
from discord.ext import commands
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import ingestion, preprocessing, processing, utils, model, news
from config import loadConfig, validateArgs

# configure Discord bot based on config
config = loadConfig()
discord_config = config.get('discord', {})

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=discord_config.get('command_prefix', '!'), intents=intents)

# load config
stock_config = config.get('default_stock', {})
default_ticker = stock_config.get('ticker', 'AAPL')
default_period = stock_config.get('period', '5y')
default_interval = stock_config.get('interval', '1d')

# load allowed args
allowed_args = config.get('allowed_arguments', {})
allowed_intervals = allowed_args.get('intervals', [])
allowed_periods = allowed_args.get('periods', [])

@bot.event
async def on_ready():
    print(f'Bot is ready. Logged in as {bot.user}')

@bot.command(name='helpStock')
async def helpStock(ctx):    
    """Show help for all stock commands"""
    help_text = """
**Chronox Stock Bot Commands**
`!fetchStock <ticker> [period] [interval]` - Fetch stock data
   - ticker: Stock symbol (e.g., AAPL)
   - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
   - interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo (default: 1d)

`!stockChart <ticker> [period] [interval]` - Get stock price chart

`!stockFeatures <ticker> [period] [interval]` - Get feature table for stock

`!predictXgboost <ticker> [period] [interval] [daysAhead] [test_size] [shuffle]` - Get XGBoost prediction
   - daysAhead: Days to predict ahead (default: 30)
   - test_size: Test size ratio (default: 0.2)
   - shuffle: Add "shuffle" to randomly shuffle data (default: False)

`!predictLightgbm <ticker> [period] [interval] [daysAhead] [test_size] [shuffle]` - Get LightGBM prediction
   - daysAhead: Days to predict ahead (default: 30)
   - test_size: Test size ratio (default: 0.2)
   - shuffle: Add "shuffle" to randomly shuffle data (default: False)

`!predictProphet <ticker> [period] [interval] [daysAhead] [test_size]` - Get Prophet prediction
   - daysAhead: Days to predict ahead (default: 30)
   - test_size: Test size ratio (default: 0.2)

`!news <ticker> [count]` - Get latest news for a ticker
   - count: Number of news items to fetch (default: 5)

`!newsEnabled <true/false>` - Toggle news sentiment analysis on/off
    """
    await ctx.send(help_text)

@bot.command(name='fetchStock')
async def fetchStockCmd(ctx, ticker=None, period=None, interval=None):
    """Fetch stock data for the given ticker"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # args validation
    valid, error_msg = validateArgs(period, interval)
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

@bot.command(name='stockChart')
async def stockChart(ctx, ticker=None, period=None, interval=None):
    """Send stock chart for the given ticker"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # args validation
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        chartPath = f'./data/raw/{ticker}_{period}_{interval}.png'
        file = discord.File(chartPath, filename=f'{ticker}_chart.png')
        await ctx.send(file=file)
    except Exception as e:
        await ctx.send(f"Error generating chart: {str(e)}")

@bot.command(name='stockFeatures')
async def stockFeatures(ctx, ticker=None, period=None, interval=None):
    """Display feature table for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # args validation
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
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

@bot.command(name='predictXgboost')
async def predictXgboost(ctx, ticker=None, period=None, interval=None, daysAhead=30, test_size=0.2, shuffle=None):
    """Get XGBoost prediction for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval
    shuffleData = shuffle == "shuffle"

    # args validation
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        daysAhead = int(daysAhead)
        test_size = float(test_size)

        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Processing data with features...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
        # prediction start
        shuffleMsg = ", with shuffled data" if shuffleData else ""
        await ctx.send(f"Running XGBoost prediction for {ticker}, {daysAhead} days ahead{shuffleMsg}...")
        result = model.trainXGBoost(data, ticker, period, interval, daysAhead, test_size, return_result=True, shuffle=shuffleData)
        
        # pred message
        predictionMsg = f"""
**XGBoost Prediction for {result['ticker']}**
- Period: {result['period']}, Interval: {result['interval']}
- Last Close Price: ${result['lastClose']:.2f}
- Predicted Price ({daysAhead} days ahead): ${result['prediction']:.2f}
- Change: ${(result['prediction'] - result['lastClose']):.2f} ({((result['prediction'] / result['lastClose'] - 1) * 100):.2f}%)
- Model Error (RMSE): {result['error']:.4f}
"""
        # extra sentiment info if enabled
        if result.get('sentimentEnabled', False):
            sentimentScore = result.get('sentimentScore', 0)
            sentiment_status = "Positive" if sentimentScore > 0.05 else "Negative" if sentimentScore < -0.05 else "Neutral"
            
            predictionMsg += f"""
**News Sentiment Analysis**
- Sentiment: {sentiment_status} ({sentimentScore:.4f})
- Original Prediction: ${result.get('originalPrediction', 0):.2f}
- Sentiment-Adjusted: ${result['prediction']:.2f}
- Adjustment: ${(result['prediction'] - result.get('originalPrediction', 0)):.2f}
"""

        await ctx.send(predictionMsg)
        
        # make chart
        chartPath = utils.generatePredictionChart(
            data=data,
            predictionValue=result['prediction'],
            days_ahead=daysAhead,
            ticker=ticker,
            period=period,
            interval=interval,
            modelName="XGBoost"
        )
        
        file = discord.File(chartPath, filename=f'{ticker}_xgboost_prediction.png')
        await ctx.send(file=file)
        
    except Exception as e:
        await ctx.send(f"Error making prediction: {str(e)}")

@bot.command(name='predictLightgbm')
async def predictLightgbm(ctx, ticker=None, period=None, interval=None, daysAhead=30, test_size=0.2, shuffle=None):
    """Get LightGBM prediction for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval
    shuffleData = shuffle == "shuffle"

    # args validation
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        daysAhead = int(daysAhead)
        test_size = float(test_size)
        
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Processing data with features...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
          # run prediction
        shuffleMsg = ", with shuffled data" if shuffleData else ""
        await ctx.send(f"Running LightGBM prediction for {ticker}, {daysAhead} days ahead{shuffleMsg}...")
        result = model.trainLightGBM(data, ticker, period, interval, daysAhead, test_size, return_result=True, shuffle=shuffleData)
        
        # output message
        predictionMsg = f"""
**LightGBM Prediction for {result['ticker']}**
- Period: {result['period']}, Interval: {result['interval']}
- Last Close Price: ${result['lastClose']:.2f}
- Predicted Price ({daysAhead} days ahead): ${result['prediction']:.2f}
- Change: ${(result['prediction'] - result['lastClose']):.2f} ({((result['prediction'] / result['lastClose'] - 1) * 100):.2f}%)
- Model Error (RMSE): {result['error']:.4f}
"""
        # sentiment if needed
        if result.get('sentimentEnabled', False):
            sentimentScore = result.get('sentimentScore', 0)
            sentiment_status = "Positive" if sentimentScore > 0.05 else "Negative" if sentimentScore < -0.05 else "Neutral"
            
            predictionMsg += f"""
**News Sentiment Analysis**
- Sentiment: {sentiment_status} ({sentimentScore:.4f})
- Original Prediction: ${result.get('originalPrediction', 0):.2f}
- Sentiment-Adjusted: ${result['prediction']:.2f}
- Adjustment: ${(result['prediction'] - result.get('originalPrediction', 0)):.2f}
"""

        await ctx.send(predictionMsg)
        
        # chart
        chartPath = utils.generatePredictionChart(
            data=data,
            predictionValue=result['prediction'],
            days_ahead=daysAhead,
            ticker=ticker,
            period=period,
            interval=interval,
            modelName="LightGBM"
        )
        
        file = discord.File(chartPath, filename=f'{ticker}_lightgbm_prediction.png')
        await ctx.send(file=file)
        
    except Exception as e:
        await ctx.send(f"Error making prediction: {str(e)}")

@bot.command(name='predictProphet')
async def predictProphet(ctx, ticker=None, period=None, interval=None, daysAhead=30, test_size=0.2):
    """Get Prophet prediction for the given stock"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # args validation
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        daysAhead = int(daysAhead)
        test_size = float(test_size)
        
        await ctx.send(f"Fetching latest {ticker} data...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Processing data with features...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
        
        # run prediction
        await ctx.send(f"Running Prophet prediction for {ticker}, {daysAhead} days ahead...")
        result = model.trainProphet(data, ticker, period, interval, daysAhead, test_size, return_result=True)
        
        # output message
        predictionMsg = f"""
**Prophet Prediction for {result['ticker']}**
- Period: {result['period']}, Interval: {result['interval']}
- Last Close Price: ${result['lastClose']:.2f}
- Predicted Price ({daysAhead} days ahead): ${result['prediction']:.2f}
- Change: ${(result['prediction'] - result['lastClose']):.2f} ({((result['prediction'] / result['lastClose'] - 1) * 100):.2f}%)
- Model Error (RMSE): {result['error']:.4f}
"""
        # sentiments
        if result.get('sentimentEnabled', False):
            sentimentScore = result.get('sentimentScore', 0)
            sentiment_status = "Positive" if sentimentScore > 0.05 else "Negative" if sentimentScore < -0.05 else "Neutral"
            
            predictionMsg += f"""
**News Sentiment Analysis**
- Sentiment: {sentiment_status} ({sentimentScore:.4f})
- Original Prediction: ${result.get('originalPrediction', 0):.2f}
- Sentiment-Adjusted: ${result['prediction']:.2f}
- Adjustment: ${(result['prediction'] - result.get('originalPrediction', 0)):.2f}
"""

        await ctx.send(predictionMsg)
        
        # chart
        chartPath = utils.generatePredictionChart(
            data=data,
            predictionValue=result['prediction'],
            days_ahead=daysAhead,
            ticker=ticker,
            period=period,
            interval=interval,
            modelName="Prophet"
        )
        
        file = discord.File(chartPath, filename=f'{ticker}_prophet_prediction.png')
        await ctx.send(file=file)
        
    except Exception as e:
        await ctx.send(f"Error making prediction: {str(e)}")

@bot.command(name='news')
async def get_news(ctx, ticker=None, count=5):
    """Get the latest news for a ticker"""
    ticker = ticker or default_ticker
    try:
        count = int(count)
        
        await ctx.send(f"Fetching latest news for {ticker}...")
        
        # get sentiment data and stories
        _, stories = news.getSentimentData(ticker)
        
        if not stories:
            await ctx.send(f"No recent news found for {ticker}")
            return 
        # stories count cap
        stories = stories[:min(count, len(stories))]
        
        # header
        if len(stories) > 1:
            await ctx.send(f"**Latest News for {ticker.upper()}** (showing {len(stories)} stories)")
        
        # if more than 1 story, multiple messages
        for i, story in enumerate(stories, 1):
            time_str = story['time'].strftime('%Y-%m-%d %H:%M:%S UTC')
            title = story.get('title', 'No title available')
            description = story.get('description', 'No description available')
            source = story.get('site', 'Unknown source')
            url = story.get('url', '')
            sentiment = news.analyze_sentiment([story])
            
            news_msg = ""
            if len(stories) == 1:
                news_msg += f"**Latest News for {ticker.upper()}**\n\n"
            
            news_msg += f"**{i}. {title}**\n"
            news_msg += f"{description}\n"
            news_msg += f"Sentiment: {sentiment['compoundAvg']:.4f}\n"
            news_msg += f"Source: {source} | {time_str}\n"
            news_msg += f"Link: {url}"
            
            # each story as separate message
            await ctx.send(news_msg)
        sentiment = news.analyze_sentiment(stories)
        await ctx.send(f"Sentiment: {sentiment['positiveRatio']:.2%} Positive | {sentiment['negativeRatio']:.2%} Negative | {sentiment['neutralRatio']:.2%} Neutral\n")
    except Exception as e:
        await ctx.send(f"Error fetching news: {str(e)}")

@bot.command(name='newsEnabled')
async def toggle_news_enabled(ctx, enabled=None):
    """Toggle news sentiment analysis on/off"""
    if enabled is None:
        await ctx.send(f"News sentiment analysis is currently {'enabled' if news.newsEnabled else 'disabled'}. Use !newsEnabled true/false to change.")
        return
        
    if enabled.lower() in ['true', '1', 'yes', 'on', 'enable']:
        news.newsEnabled = True
        await ctx.send("✅ News sentiment analysis has been enabled.")
    elif enabled.lower() in ['false', '0', 'no', 'off', 'disable']:
        news.newsEnabled = False
        await ctx.send("❌ News sentiment analysis has been disabled.")
    else:
        await ctx.send("Invalid option. Please use true/false, on/off, enable/disable, yes/no, or 1/0.")

@bot.command(name='prune')
async def prune(ctx, count="5"):
    try:
        count = int(count)
        if count <= 0:
            await ctx.send("Please provide a positive number of messages to delete.")
            return
            
        # get recent messages
        messages = []
        async for message in ctx.channel.history(limit=100):
            messages.append(message)
        
        # only get bot messages
        bot_messages = [message for message in messages if message.author == bot.user]
        
        # take count of mesages 
        messages_to_delete = bot_messages[:min(count, len(bot_messages))]
        
        # deletion
        deleted_count = 0
        for message in messages_to_delete:
            await message.delete()
            deleted_count += 1
            
        confirmation_msg = await ctx.send(f"Deleted {deleted_count} bot messages.")
        
        # delete the command mesage
        await ctx.message.delete()
        
    except Exception as e:
        await ctx.send(f"Error pruning messages: {str(e)}")

def runDiscordBot(token=None):
    """Run the Discord bot with the given token or from config"""
    if token is None:
        token = discord_config.get('token', os.getenv("DISCORD_TOKEN"))
    
    if not token:
        print("No Discord token found in config or environment variables")
        return
        
    bot.run(token)

if __name__ == "__main__":
    # try grab token from config, fallback to env var if not found
    token = discord_config.get('token')
    if not token:
        token = os.getenv("CHRONOX_DISCORD_TOKEN")
        
    if not token:
        print("Please set a Discord token in config.json or DISCORD_TOKEN environment variable")
    else:
        runDiscordBot(token)
