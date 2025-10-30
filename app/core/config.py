import os


class Settings:
    model_id: str = os.getenv("MODEL_ID", "cedrugs/toxic-xlmr")
    device: str = os.getenv("DEVICE", "").strip().lower()
    max_length: int = int(os.getenv("MAX_LENGTH", "256"))
    title: str = "Chatguard API"
    version: str = "1.0.0"


settings = Settings()
