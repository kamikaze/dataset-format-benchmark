from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    dummy_param: Optional[str] = None


settings = Settings()
