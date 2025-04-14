# Chronox Stock Predictor Bot

Chronox is a tool that fetches stock data, processes it, predicts future prices using machine learning models, and presents the results via a Discord bot.

## Key Features

*   Fetches historical stock data using yfinance.
*   Calculates various technical indicators (like MA, EMA, MACD, RSI, Bollinger Bands).
*   Trains ML models (XGBoost, Prophet) to predict future stock prices.
*   Integrates with a Discord bot for easy interaction:
    *   Fetch data (`!fetchStock`)
    *   Generate predictions (`!predict[Xgboost|Prophet|Lightgbm]`)
    *   Display calculated features (`!stockFeatures`)
    *   Show prediction charts (`!chart`)
    *   For more commands use `!help`
*   Includes sentiment analysis from news headlines to potentially adjust predictions (`!newsEnabled`).
*   Offers data cleanup capabilities.

## Requirements

*   Python 3.x
*   Dependencies listed in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
*   A Discord Bot Token.

## How to Run

1.  **Configuration**:
    *   Make sure you have a Discord bot token.
    *   You can set the token in a `config.json` file (create one if it doesn't exist) like this:
        ```json
        {
          "discord": {
            "token": "YOUR_DISCORD_BOT_TOKEN"
          }
        }
        ```
    *   Alternatively, set the `CHRONOX_DISCORD_TOKEN` environment variable.

2.  **Run the main script**:
    ```bash
    python src/main.py
    ```
    This will start the Discord bot.

## Docker

A `Dockerfile` is included in the project. You can build and run the application in a container:

1.  **Build the image**:
    ```bash
    docker build -t chronox .
    ```
2.  **Run the container** (make sure to pass your Discord token):
    ```bash
    docker run -e CHRONOX_DISCORD_TOKEN="YOUR_DISCORD_BOT_TOKEN" chronox
    ```
You can find prebuilt images on Dockerhub: https://hub.docker.com/repository/docker/jakstein/chronox

[![image.png](https://i.postimg.cc/zfytR1xg/image.png)](https://postimg.cc/mcGN0KfL)
## License
This project is licensed under the MIT License.