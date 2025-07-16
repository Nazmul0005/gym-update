import os
from dotenv import load_dotenv
            
load_dotenv()

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.openai_api_key = os.getenv("OPENAI_API_KEY")
            cls._instance.model_name = os.getenv("MODEL")
            cls._instance.model_name2 = os.getenv("MODEL2")
            cls._instance.tavily_api_key = os.getenv("TAVILY_API_KEY")

        return cls._instance