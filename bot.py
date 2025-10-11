# bot.py
import os
import re
import asyncio
from typing import List, Tuple, Optional

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message

from yandex_music import Client
from yandex_music.exceptions import YandexMusicError

from openai import OpenAI

from db import upsert_user, log_event, total_users, active_since

import logging
logging.basicConfig(level=logging.INFO)


# ==============================
# Configuration
# ==============================
load_dotenv()  # читаем .env при локальном запуске / внутри контейнера

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("MODEL", "").strip() or None
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").replace(" ", "").split(",") if x.isdigit()}

from handlers.help import router as help_router
from handlers.ads  import router as ads_router

# SOCKS-прокси для Яндекс Музыки (внутри docker-compose это socks:9050)
YM_PROXY_HTTP = os.getenv("YM_PROXY_HTTP", "").strip()   # напр. socks5h://socks:9050
YM_PROXY_HTTPS = os.getenv("YM_PROXY_HTTPS", "").strip() # напр. socks5h://socks:9050

# Таймауты/ретраи для ЯМ (используются библиотекой requests внутри yandex_music)
YM_TIMEOUT = float(os.getenv("YM_TIMEOUT", "15"))
# *Ретраи/backoff можно реализовать снаружи при желании

# Превью (сколько строк показать «для красоты», это НЕ влияет на анализ)
PREVIEW_COUNT = int(os.getenv("PREVIEW_COUNT", "30"))

# Чанкинг для анализа: уходит вся совокупность треков
PROMPT_CHUNK_SIZE = int(os.getenv("PROMPT_CHUNK_SIZE", "220"))
PROMPT_MAX_CHUNKS = int(os.getenv("PROMPT_MAX_CHUNKS", "20"))  # страховка бюджета

# Отправка длинного ответа
MAX_TG = 4000  # безопасный размер куска сообщения (чуть меньше 4096)



# ==============================
# Utilities
# ==============================

def split_text_for_tg(s: str, limit: int = MAX_TG) -> List[str]:
    """Режем длинный текст на части ≤ limit, стараясь резать по строкам."""
    parts, cur, size = [], [], 0
    for line in s.splitlines(True):  # сохраняем \n
        if size + len(line) > limit and cur:
            parts.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        parts.append("".join(cur))
    return parts


async def send_long(m: Message, text: str, parse_mode: Optional[str] = None):
    """
    Отправляет длинный текст несколькими сообщениями.
    По умолчанию parse_mode=None (чтобы исключить ошибки парсинга из-за «сырых» символов).
    """
    for chunk in split_text_for_tg(text):
        await m.answer(chunk, parse_mode=parse_mode)


def parse_ym_playlist_url(url: str) -> Optional[Tuple[str, int]]:
    """
    Ожидаем ссылки вида:
    https://music.yandex.ru/users/<user>/playlists/<kind>
    Возвращает (user, kind) или None
    """
    m = re.search(r"music\.yandex\.(?:ru|com)/users/([^/]+)/playlists/(\d+)", url)
    if not m:
        return None
    user = m.group(1)
    kind = int(m.group(2))
    return user, kind


def make_client(auth: bool = False) -> Client:
    """
    Создаёт клиент ЯМ. Для публичных плейлистов токен не обязателен.
    Если понадобится — можно читать YM_TOKEN из окружения.
    """
    token = os.getenv("YM_TOKEN", "").strip() if auth else None
    client = Client(token) if token else Client()
    # библиотека на базе requests; таймаут пробросим через session при наличии
    try:
        session = getattr(client, "_session", None) or getattr(client, "session", None)
        if session:
            session.timeout = YM_TIMEOUT
            # если у requests.Session есть proxies, пробросим туда SOCKS
            if hasattr(session, "proxies"):
                proxies = {}
                if YM_PROXY_HTTP:
                    proxies["http"] = YM_PROXY_HTTP
                if YM_PROXY_HTTPS:
                    proxies["https"] = YM_PROXY_HTTPS
                if proxies:
                    session.proxies.update(proxies)
    except Exception:
        pass
    return client


def fetch_tracks(user: str, kind: int) -> List[Tuple[str, str]]:
    """
    Возвращает ВСЕ треки [(artist, title), ...] из плейлиста.
    На время запросов к ЯМ принудительно включаем HTTP(S)_PROXY на основе YM_PROXY_*,
    чтобы любой внутренний requests шёл через SOCKS (RUS IP).
    """
    last_err = None

    proxy_http = YM_PROXY_HTTP
    proxy_https = YM_PROXY_HTTPS
    env_http_proxy = proxy_http or proxy_https or None
    env_https_proxy = proxy_https or proxy_http or None

    old_http_proxy = os.environ.get("HTTP_PROXY")
    old_https_proxy = os.environ.get("HTTPS_PROXY")
    if env_http_proxy:
        os.environ["HTTP_PROXY"] = env_http_proxy
    if env_https_proxy:
        os.environ["HTTPS_PROXY"] = env_https_proxy

    try:
        for attempt in (False, True):  # False=anonymous, True=with token
            try:
                client = make_client(auth=attempt)
                playlist = client.users_playlists(kind=kind, user_id=user)
                tracks: List[Tuple[str, str]] = []
                for pl_tr in (playlist.tracks or []):
                    tr = getattr(pl_tr, "track", None)
                    if not tr:
                        continue
                    title = (getattr(tr, "title", "") or "").strip() or "Untitled"
                    artists = ", ".join(a.name for a in (getattr(tr, "artists", None) or []) if a and a.name) or "Unknown artist"
                    tracks.append((artists, title))
                return tracks
            except YandexMusicError as e:
                msg = str(e)
                # Главная проблема ранее — 451
                if "Unavailable For Legal Reasons" in msg or "451" in msg:
                    last_err = RuntimeError(
                        "Яндекс вернул 451 (гео-ограничения). "
                        "Проверьте, что SOCKS-прокси действительно российский и доступен из контейнера (socks:9050)."
                    )
                    break
                last_err = e
                # пробуем со вторым способом (с токеном), если первый не прошёл
                continue
            except Exception as e:
                last_err = e
                break
    finally:
        # откатываем системные прокси, чтобы OpenAI шёл напрямую из ЕС
        if old_http_proxy is None:
            os.environ.pop("HTTP_PROXY", None)
        else:
            os.environ["HTTP_PROXY"] = old_http_proxy
        if old_https_proxy is None:
            os.environ.pop("HTTPS_PROXY", None)
        else:
            os.environ["HTTPS_PROXY"] = old_https_proxy

    raise RuntimeError(f"Не удалось получить плейлист: {last_err}")


# ==============================
# LLM prompts & chunking
# ==============================

def tracks_to_bullets(tracks: List[Tuple[str, str]]) -> str:
    return "\n".join(f"- {a} — {t}" for a, t in tracks)


def build_chunk_messages(k: int, n: int, chunk: List[Tuple[str, str]]) -> List[dict]:
    bullets = tracks_to_bullets(chunk)
    content = (
        f"ЧАСТЬ {k}/{n}. Ниже подсписок треков из большого плейлиста.\n\n"
        f"{bullets}\n\n"
        "ЗАДАЧА: извлеки агрегированные признаки по этой части (жанры/поджанры, темпы/энергичность, настроения, темы текстов, языки; "
        "прокси Big Five на основе ЭТОЙ ЧАСТИ).\n"
        "Формат: краткая сводка (проценты/доли, если уместно), затем нумерованные тезисы по Big Five для ЭТОЙ ЧАСТИ.\n"
        "Не делай финальных выводов по всему плейлисту."
    )
    return [
        {"role": "system",
         "content": ("Ты — специалист по музыкальной психографии, работаешь в парадигме Big Five. "
                     "Оперируй эмпирическими связями жанров/темпов/настроений/текста с чертами личности "
                     "(см. Rentfrow & Gosling, 2003; Greenberg et al., 2016)."
                     "Не используй спецсимволы разметки (#, ##, **, *), можно умеренно использовать эмодзи")},
        {"role": "user", "content": content},
    ]


def build_synthesis_messages(partials: List[str]) -> List[dict]:
    joined = "\n\n".join(f"=== ЧАСТЬ {i+1} ===\n{p}" for i, p in enumerate(partials))
    content = (
        "Ниже — частичные отчёты по частям одного большого плейлиста.\n"
        "Объедини их и сделай финальный психографический анализ по всей совокупности треков.\n"
        "Структура итогового ответа:\n"
        "1) Профиль Big Five (с обоснованиями и индикаторами)\n"
        "2) Эмоции/внутренние состояния\n"
        "3) Функции плейлиста (саморегуляция, бегство, вдохновение, идентичность и т.п.)\n"
        "4) Психологический портрет\n\n"
        "Исходные частичные отчёты:\n"
        f"{joined}\n"
    )
    return [
        {"role": "system",
         "content": ("Ты — специалист по музыкальной психографии (Big Five). Делай выводы аккуратно, "
                     "ссылаясь на паттерны жанров, темпов, настроений и семантики.")},
        {"role": "user", "content": content},
    ]


def openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def pick_model_sequence() -> List[str]:
    seq: List[str] = []
    if MODEL:
        seq.append(MODEL)
    # фоллбэки по возрастанию «универсальности»
    seq += ["gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"]
    # уберём дубликаты, сохраняя порядок
    seen, out = set(), []
    for m in seq:
        if m and m not in seen:
            out.append(m); seen.add(m)
    return out


def chat_complete(messages: List[dict], temperature: float = 0.7, max_tokens: int = 900) -> str:
    cli = openai_client()
    last_err = None
    for m in pick_model_sequence():
        try:
            r = cli.chat.completions.create(model=m, messages=messages,
                                            temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            continue
    # если ни одна модель не сработала
    raise RuntimeError(f"OpenAI недоступен или модель запрещена: {last_err}")


def analyze_full_playlist_all_tracks(tracks: List[Tuple[str, str]]) -> str:
    """
    Полный анализ ВСЕХ треков через чанки + финальную синтезу.
    """
    if not OPENAI_API_KEY:
        return ("Аналитика отключена: нет OPENAI_API_KEY.\n"
                "Треки получены, но для психографического анализа нужен ключ OpenAI.")

    if not tracks:
        return "Плейлист пуст."

    # 1) Чанки
    chunks: List[List[Tuple[str, str]]] = [
        tracks[i:i + PROMPT_CHUNK_SIZE] for i in range(0, len(tracks), PROMPT_CHUNK_SIZE)
    ]
    if len(chunks) > PROMPT_MAX_CHUNKS:
        chunks = chunks[:PROMPT_MAX_CHUNKS]  # страхуем бюджет

    partials: List[str] = []
    n = len(chunks)
    for i, ch in enumerate(chunks, 1):
        msgs = build_chunk_messages(i, n, ch)
        part = chat_complete(msgs, temperature=0.7, max_tokens=900)
        partials.append(part)

    # 2) Финальная синтеза
    final = chat_complete(build_synthesis_messages(partials), temperature=0.6, max_tokens=1400)
    return final


# ==============================
# Telegram bot (Aiogram 3.12)
# ==============================

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=None),  # по умолчанию без Markdown, чтобы не ловить parse errors
)
dp = Dispatcher()

# Подключаем роутеры
dp.include_router(help_router)
dp.include_router(ads_router)



@dp.message(CommandStart())
async def on_start(m: Message):
    upsert_user(m.from_user); log_event(m.from_user.id, "start")
    await m.answer(
        "Привет! Пришли публичную ссылку на плейлист Яндекс.Музыки вида:\n"
        "https://music.yandex.ru/users/<user>/playlists/<kind>\n\n"
        "Я получу все треки и сделаю психографический анализ (Big Five)."
    )


@dp.message(F.text.contains("music.yandex."))
async def on_text(m: Message):
    url = m.text.strip()
    parsed = parse_ym_playlist_url(url)
    if not parsed:
        await m.answer("Не смог распознать ссылку на плейлист. Нужен формат: https://music.yandex.ru/users/<user>/playlists/<kind>")
        return

    user, kind = parsed
    await m.answer("Получаю треки, подождите...")

    try:
        # 1) Забираем ВСЕ треки
        tracks = await asyncio.to_thread(fetch_tracks, user, kind)
        total = len(tracks)

        # 2) Красивое превью (не влияет на анализ)
        preview_lines = []
        for i, (a, t) in enumerate(tracks[:PREVIEW_COUNT], 1):
            preview_lines.append(f"{i}. {a} — {t}")
        preview_text = "\n".join(preview_lines)
        if total > PREVIEW_COUNT:
            preview_text += f"\n… и ещё {total - PREVIEW_COUNT} трек(ов)."

        await send_long(
            m,
            f"Найдено треков: {total}\n\nПревью:\n{preview_text}",
            parse_mode=None
        )

        # 3) Запускаем ПОЛНЫЙ анализ по чанкам
        await m.answer("Минуту, запускаю психографический анализ...")
        analysis = await asyncio.to_thread(analyze_full_playlist_all_tracks, tracks)
        log_event(m.from_user.id, "analysis_done")

        # 4) Отправляем полный ответ (в нескольких сообщениях при необходимости)
        await send_long(m, analysis, parse_mode=None)

    except Exception as e:
        # Ошибки отправляем без markdown, чтобы исключить парсинг
        await m.answer(f"Ошибка: {e}", parse_mode=None)

@dp.message(F.text == "/stats")
async def stats(m: Message):
    if ADMIN_IDS and m.from_user.id not in ADMIN_IDS:
        return
    upsert_user(m.from_user); log_event(m.from_user.id, "stats")
    total = total_users()
    d1 = active_since(24*3600)
    d7 = active_since(7*24*3600)
    d30 = active_since(30*24*3600)
    await m.answer(
        f"Users total: {total}\nActive 24h: {d1}\nActive 7d: {d7}\nActive 30d: {d30}",
        parse_mode=None
    )

async def main():
    if not BOT_TOKEN:
        print("BOT_TOKEN не задан в окружении")
        return
    
    from aiogram.types import BotCommand
    await bot.set_my_commands([
        BotCommand(command="start",   description="Начать"),
        BotCommand(command="help",    description="Как пользоваться"),
        BotCommand(command="reklama", description="Реклама и партнёрство"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
