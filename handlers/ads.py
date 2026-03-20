import os
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message

from db import upsert_user, log_event

router = Router()

# Можно вынести в config.py, но для простоты читаем из .env прямо тут
AD_CONTACT = os.getenv("AD_CONTACT", "@your_contact")
AD_TEXT = os.getenv(
    "AD_TEXT",
    "Реклама и партнёрства: нативные интеграции, спецпроекты, white-label.\nСвязь: {contact}"
)
AD_LINK = os.getenv("AD_LINK", "")  # можно указать лендинг/форму


def build_ad_message() -> str:
    text = AD_TEXT.format(contact=AD_CONTACT)
    if AD_LINK:
        text += f"\nПодробнее: {AD_LINK}"
    return text

@router.message(Command("reklama"))
async def on_reklama(m: Message):
    upsert_user(m.from_user)
    log_event(m.from_user.id, "ad")
    await m.answer(build_ad_message(), parse_mode=None)

# Кириллический «алиас»: Telegram-команды официально латиницей,
# поэтому ловим текстом "/реклама" или просто "реклама".
@router.message(F.text.in_({"/реклама", "реклама"}))
async def on_reklama_cyr(m: Message):
    await on_reklama(m)
