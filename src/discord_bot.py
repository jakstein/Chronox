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
    print(f'Bot jest gotowy. Zalogowano jako {bot.user}')

@bot.command(name='helpStock')
async def helpStock(ctx):    
    """Pokaż pomoc dla wszystkich poleceń giełdowych"""
    help_text = """
**Polecenia Bota Chronox Stock**
`!fetchStock <ticker> [period] [interval]` - Pobierz dane akcji
   - ticker: Symbol akcji (np. AAPL)
   - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (domyślnie: 1y)
   - interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo (domyślnie: 1d)

`!stockChart <ticker> [period] [interval]` - Uzyskaj wykres cen akcji

`!stockFeatures <ticker> [period] [interval]` - Uzyskaj tabelę cech dla akcji

`!predictXgboost <ticker> [period] [interval] [daysAhead] [test_size] [shuffle]` - Uzyskaj prognozę XGBoost
   - daysAhead: Liczba dni do przewidzenia (domyślnie: 30)
   - test_size: Współczynnik wielkości testu (domyślnie: 0.2)
   - shuffle: Dodaj "shuffle", aby losowo wymieszać dane (domyślnie: False)

`!predictLightgbm <ticker> [period] [interval] [daysAhead] [test_size] [shuffle]` - Uzyskaj prognozę LightGBM
   - daysAhead: Liczba dni do przewidzenia (domyślnie: 30)
   - test_size: Współczynnik wielkości testu (domyślnie: 0.2)
   - shuffle: Dodaj "shuffle", aby losowo wymieszać dane (domyślnie: False)

`!predictProphet <ticker> [period] [interval] [daysAhead] [test_size]` - Uzyskaj prognozę Prophet
   - daysAhead: Liczba dni do przewidzenia (domyślnie: 30)
   - test_size: Współczynnik wielkości testu (domyślnie: 0.2)

`!news <ticker> [count]` - Uzyskaj najnowsze wiadomości dla akcji
   - count: Liczba wiadomości do pobrania (domyślnie: 5)
    """
    await ctx.send(help_text)

@bot.command(name='fetchStock')
async def fetchStockCmd(ctx, ticker=None, period=None, interval=None):
    """Pobierz dane akcji dla podanego tickera"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # walidacja argumentów
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    await ctx.send(f"Pobieranie danych akcji {ticker} dla okresu {period} z interwałem {interval}...")
    
    try:
        os.makedirs('./data/raw', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        
        ingestion.fetchStock(ticker, period, interval)
        await ctx.send(f"Pomyślnie pobrano dane dla {ticker}!")
        
        file = discord.File(f'./data/raw/{ticker}_{period}_{interval}.png', filename=f'{ticker}_chart.png')
        await ctx.send(file=file)
    except Exception as e:
        await ctx.send(f"Błąd podczas pobierania danych akcji: {str(e)}")

@bot.command(name='stockChart')
async def stockChart(ctx, ticker=None, period=None, interval=None):
    """Wyślij wykres akcji dla podanego tickera"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # walidacja argumentów
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        await ctx.send(f"Pobieranie najnowszych danych dla {ticker}...")
        ingestion.fetchStock(ticker, period, interval)
        
        chartPath = f'./data/raw/{ticker}_{period}_{interval}.png'
        file = discord.File(chartPath, filename=f'{ticker}_chart.png')
        await ctx.send(file=file)
    except Exception as e:
        await ctx.send(f"Błąd podczas generowania wykresu: {str(e)}")

@bot.command(name='stockFeatures')
async def stockFeatures(ctx, ticker=None, period=None, interval=None):
    """Wyświetl tabelę cech dla danej akcji"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # walidacja argumentów
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        await ctx.send(f"Pobieranie najnowszych danych dla {ticker}...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Obliczanie cech...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
    
        data = pd.read_csv(processed_path)
        
        # ostatnie 10 wierszy do wyświetlenia
        last_rows = data.tail(10)
        
        # formatowanie jako tabela tekstowa dla Discord
        features_str = f"**Cechy {ticker} (ostatnie 10 wierszy)**\n```\n"
        features_str += last_rows.to_string(index=False)
        features_str += "\n```"
        
        if len(features_str) > 2000:
            # w przypadku zbyt długiej wiadomości dla Discord, wyślij jako plik CSV
            buffer = io.StringIO()
            last_rows.to_csv(buffer, index=False)
            buffer.seek(0)
            
            file = discord.File(
                fp=io.BytesIO(buffer.getvalue().encode()),
                filename=f"{ticker}_features.csv"
            )
            await ctx.send("Tabela cech (ostatnie 10 wierszy):", file=file)
        else:
            await ctx.send(features_str)
            
    except Exception as e:
        await ctx.send(f"Błąd podczas generowania cech: {str(e)}")

@bot.command(name='predictXgboost')
async def predictXgboost(ctx, ticker=None, period=None, interval=None, daysAhead=30, test_size=0.2, shuffle=None):
    """Uzyskaj prognozę XGBoost dla danej akcji"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval
    shuffleData = shuffle == "shuffle"

    # walidacja argumentów
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        daysAhead = int(daysAhead)
        test_size = float(test_size)

        await ctx.send(f"Pobieranie najnowszych danych dla {ticker}...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Przetwarzanie danych z cechami...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
        # rozpoczęcie prognozy
        shuffleMsg = ", z przemieszanymi danymi" if shuffleData else ""
        await ctx.send(f"Uruchamianie prognozy XGBoost dla {ticker}, {daysAhead} dni do przodu{shuffleMsg}...")
        result = model.trainXGBoost(data, ticker, period, interval, daysAhead, test_size, return_result=True, shuffle=shuffleData)
        
        # wiadomość z prognozą
        predictionMsg = f"""
**Prognoza XGBoost dla {result['ticker']}**
- Okres: {result['period']}, Interwał: {result['interval']}
- Ostatnia cena zamknięcia: ${result['lastClose']:.2f}
- Prognozowana cena (za {daysAhead} dni): ${result['prediction']:.2f}
- Zmiana: ${(result['prediction'] - result['lastClose']):.2f} ({((result['prediction'] / result['lastClose'] - 1) * 100):.2f}%)
- Błąd modelu (RMSE): {result['error']:.4f}
"""        # dodatkowe informacje o sentymencie, jeśli włączone
        if result.get('sentimentEnabled', False):
            sentimentScore = result.get('sentimentScore', 0)
            sentiment_status = "Pozytywny" if sentimentScore > 0.05 else "Negatywny" if sentimentScore < -0.05 else "Neutralny"
            
            predictionMsg += f"""
**Analiza Sentymentu Wiadomości**
- Sentyment: {sentiment_status} ({sentimentScore:.4f})
- Oryginalna prognoza: ${result.get('originalPrediction', 0):.2f}
- Po korekcie sentymentu: ${result['prediction']:.2f}
- Korekta: ${(result['prediction'] - result.get('originalPrediction', 0)):.2f}
"""

        await ctx.send(predictionMsg)
        
        # tworzenie wykresu
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
        await ctx.send(f"Błąd podczas tworzenia prognozy: {str(e)}")

@bot.command(name='predictLightgbm')
async def predictLightgbm(ctx, ticker=None, period=None, interval=None, daysAhead=30, test_size=0.2, shuffle=None):
    """Uzyskaj prognozę LightGBM dla danej akcji"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval
    shuffleData = shuffle == "shuffle"

    # walidacja argumentów
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        daysAhead = int(daysAhead)
        test_size = float(test_size)
        
        await ctx.send(f"Pobieranie najnowszych danych dla {ticker}...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Przetwarzanie danych z cechami...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
          # uruchomienie prognozy
        shuffleMsg = ", z przemieszanymi danymi" if shuffleData else ""
        await ctx.send(f"Uruchamianie prognozy LightGBM dla {ticker}, {daysAhead} dni do przodu{shuffleMsg}...")
        result = model.trainLightGBM(data, ticker, period, interval, daysAhead, test_size, return_result=True, shuffle=shuffleData)
        
        # wiadomość z wynikiem
        predictionMsg = f"""
**Prognoza LightGBM dla {result['ticker']}**
- Okres: {result['period']}, Interwał: {result['interval']}
- Ostatnia cena zamknięcia: ${result['lastClose']:.2f}
- Prognozowana cena (za {daysAhead} dni): ${result['prediction']:.2f}
- Zmiana: ${(result['prediction'] - result['lastClose']):.2f} ({((result['prediction'] / result['lastClose'] - 1) * 100):.2f}%)
- Błąd modelu (RMSE): {result['error']:.4f}
"""        # analiza sentymentu jeśli potrzebna
        if result.get('sentimentEnabled', False):
            sentimentScore = result.get('sentimentScore', 0)
            sentiment_status = "Pozytywny" if sentimentScore > 0.05 else "Negatywny" if sentimentScore < -0.05 else "Neutralny"
            
            predictionMsg += f"""
**Analiza Sentymentu Wiadomości**
- Sentyment: {sentiment_status} ({sentimentScore:.4f})
- Oryginalna prognoza: ${result.get('originalPrediction', 0):.2f}
- Po korekcie sentymentu: ${result['prediction']:.2f}
- Korekta: ${(result['prediction'] - result.get('originalPrediction', 0)):.2f}
"""

        await ctx.send(predictionMsg)
        
        # wykres
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
        await ctx.send(f"Błąd podczas tworzenia prognozy: {str(e)}")

@bot.command(name='predictProphet')
async def predictProphet(ctx, ticker=None, period=None, interval=None, daysAhead=30, test_size=0.2):
    """Uzyskaj prognozę Prophet dla danej akcji"""
    ticker = ticker or default_ticker
    period = period or default_period
    interval = interval or default_interval

    # walidacja argumentów
    valid, error_msg = validateArgs(period, interval)
    if not valid:
        await ctx.send(f"❌ {error_msg}")
        return

    try:
        daysAhead = int(daysAhead)
        test_size = float(test_size)
        
        await ctx.send(f"Pobieranie najnowszych danych dla {ticker}...")
        ingestion.fetchStock(ticker, period, interval)
        
        raw_path = f'./data/raw/{ticker}_{period}_{interval}.csv'
        processed_path = f'./data/processed/{ticker}_{period}_{interval}.csv'
        
        await ctx.send("Przetwarzanie danych z cechami...")
        data = utils.loadData(raw_path)
        data = preprocessing.cleanData(data, ticker, period, interval)
        data = processing.addFeatures(data, ticker, period, interval)
        
        data = pd.read_csv(processed_path)
        
        # uruchomienie prognozy
        await ctx.send(f"Uruchamianie prognozy Prophet dla {ticker}, {daysAhead} dni do przodu...")
        result = model.trainProphet(data, ticker, period, interval, daysAhead, test_size, return_result=True)
        
        # wiadomość wynikowa
        predictionMsg = f"""
**Prognoza Prophet dla {result['ticker']}**
- Okres: {result['period']}, Interwał: {result['interval']}
- Ostatnia cena zamknięcia: ${result['lastClose']:.2f}
- Prognozowana cena (za {daysAhead} dni): ${result['prediction']:.2f}
- Zmiana: ${(result['prediction'] - result['lastClose']):.2f} ({((result['prediction'] / result['lastClose'] - 1) * 100):.2f}%)
- Błąd modelu (RMSE): {result['error']:.4f}
"""        # analiza sentymentu
        if result.get('sentimentEnabled', False):
            sentimentScore = result.get('sentimentScore', 0)
            sentiment_status = "Pozytywny" if sentimentScore > 0.05 else "Negatywny" if sentimentScore < -0.05 else "Neutralny"
            
            predictionMsg += f"""
**Analiza Sentymentu Wiadomości**
- Sentyment: {sentiment_status} ({sentimentScore:.4f})
- Oryginalna prognoza: ${result.get('originalPrediction', 0):.2f}
- Po korekcie sentymentu: ${result['prediction']:.2f}
- Korekta: ${(result['prediction'] - result.get('originalPrediction', 0)):.2f}
"""

        await ctx.send(predictionMsg)
        
        # wykres
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
        await ctx.send(f"Błąd podczas tworzenia prognozy: {str(e)}")

@bot.command(name='news')
async def get_news(ctx, ticker=None, count=5):
    """Pobierz najnowsze wiadomości dla danego tickera"""
    ticker = ticker or default_ticker
    try:
        count = int(count)
        
        await ctx.send(f"Pobieranie najnowszych wiadomości dla {ticker}...")
        
        # pobierz dane sentymentu i wiadomości
        _, stories = news.getSentimentData(ticker)
        
        if not stories:
            await ctx.send(f"Nie znaleziono ostatnich wiadomości dla {ticker}")
            return
            
        # ogranicz liczbę wiadomości
        stories = stories[:min(count, len(stories))]
        
        news_msg = f"**Najnowsze wiadomości dla {ticker.upper()}**\n\n"
        
        # if more than 1 story, multiple messages
        for i, story in enumerate(stories, 1):
            time_str = story['time'].strftime('%Y-%m-%d %H:%M:%S UTC')
            title = story.get('title', 'Brak dostępnego tytułu')
            description = story.get('description', 'Brak dostępnego opisu')
            source = story.get('site', 'Nieznane źródło')
            url = story.get('url', '')
            sentiment = news.analyze_sentiment([story])
            
            news_msg = ""
            if len(stories) == 1:
                news_msg += f"**Latest News for {ticker.upper()}**\n\n"
            
            news_msg += f"**{i}. {title}**\n"
            news_msg += f"{description}\n"
            news_msg += f"Sentyment: {sentiment['compoundAvg']:.4f}\n"
            news_msg += f"Źródło: {source} | {time_str}\n"
            news_msg += f"Link: {url}"
            
            # limit znaków Discorda (2000)
            if len(news_msg) > 1800 and i < len(stories):
                await ctx.send(news_msg)
                news_msg = ""
        
        if news_msg:
            await ctx.send(news_msg)
        sentiment = news.analyze_sentiment(stories)
        await ctx.send(f"Sentyment: {sentiment['positiveRatio']:.2%} Pozytywny | {sentiment['negativeRatio']:.2%} Negatywny | {sentiment['neutralRatio']:.2%} Neutralny\n")
    except Exception as e:
        await ctx.send(f"Błąd podczas pobierania wiadomości: {str(e)}")
@bot.command(name='prune')
async def prune(ctx, count="5"):
    try:
        count = int(count)
        if count <= 0:
            await ctx.send("Podaj dodatnią liczbę wiadomości do usunięcia.")
            return
            
        # pobierz ostatnie wiadomości
        messages = []
        async for message in ctx.channel.history(limit=100):
            messages.append(message)
        
        # tylko wiadomości bota
        bot_messages = [message for message in messages if message.author == bot.user]
        
        # weź określoną liczbę wiadomości 
        messages_to_delete = bot_messages[:min(count, len(bot_messages))]
        
        # usuwanie
        deleted_count = 0
        for message in messages_to_delete:
            await message.delete()
            deleted_count += 1
            
        confirmation_msg = await ctx.send(f"Usunięto {deleted_count} wiadomości bota.")
        
        # usuń komendę
        await ctx.message.delete()
        
    except Exception as e:
        await ctx.send(f"Błąd podczas usuwania wiadomości: {str(e)}")

def runDiscordBot(token=None):
    """Uruchom bota Discord z podanym tokenem lub z konfiguracji"""
    if token is None:
        token = discord_config.get('token', os.getenv("DISCORD_TOKEN"))
    
    if not token:
        print("Nie znaleziono tokenu Discord w konfiguracji ani zmiennych środowiskowych")
        return
        
    bot.run(token)

if __name__ == "__main__":
    # spróbuj pobrać token z konfiguracji, w razie niepowodzenia użyj zmiennej środowiskowej
    token = discord_config.get('token')
    if not token:
        token = os.getenv("DISCORD_TOKEN")
        
    if not token:
        print("Ustaw token Discord w pliku config.json lub zmiennej środowiskowej DISCORD_TOKEN")
    else:
        runDiscordBot(token)
