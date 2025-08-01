from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Environment setting: 'prod', 'dev', 'test'
    ENV: str = os.getenv("ENV", "prod")

    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

    # API Keys for different services
    # It's recommended to load these from environment variables for security
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "your_finnhub_api_key_here")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "your_alpaca_api_key_here")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "your_alpaca_secret_key_here")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "your_news_api_key_here")

    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        # This allows loading variables from a .env file, useful for local development
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
