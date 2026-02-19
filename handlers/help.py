from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from db import upsert_user, log_event

router = Router()

HELP_TEXT = (
    "🧭 Помощь\n\n"
    "Нажмите /start и выберите сценарий через кнопки:\n"
    "• Мой музыкальный портрет\n"
    "• Совместимость с близким\n\n"
    "Далее просто отправьте ссылку(и) в формате:\n"
    "https://music.yandex.ru/users/<user>/playlists/<kind>\n\n"
    "В режиме совместимости бот подсветит общие треки,\n"
    "оценит близость вкусов и соберёт совместимый психологический профиль пары.\n\n"
    "В личном режиме бот пришлёт отчёт: Big Five, эмоции, функции плейлиста, портрет.\n"
    "Важно: работают только публичные плейлисты; для точности нужно ≥10 треков.\n\n"
    "/start — начать\n"
    "/help — эта справка\n"
    "/reklama — партнёрство и размещение\n"
)

@router.message(Command("help"))
async def on_help(m: Message):
    upsert_user(m.from_user)
    log_event(m.from_user.id, "help")
    await m.answer(HELP_TEXT, parse_mode=None)
