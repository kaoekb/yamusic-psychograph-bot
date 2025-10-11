from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from db import upsert_user, log_event

router = Router()

HELP_TEXT = (
    "🧭 Помощь\n\n"
    "Пришлите публичную ссылку на плейлист Яндекс.Музыки в формате:\n"
    "https://music.yandex.ru/users/<user>/playlists/<kind>\n\n"
    "Бот загрузит треки и пришлёт отчёт: Big Five, эмоции, функции плейлиста, портрет.\n"
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
