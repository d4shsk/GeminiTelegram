from .chat import router as chat_router
from .photo import router as photo_router
from .settings import router as settings_router
from .start import router as start_router

all_routers = [start_router, settings_router, photo_router, chat_router]
