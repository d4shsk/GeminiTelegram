from datetime import datetime

import pytz


def moscow_datetime() -> str:
    timezone = pytz.timezone("Europe/Moscow")
    return datetime.now(timezone).strftime("%d %B %Y, %H:%M МСК")
