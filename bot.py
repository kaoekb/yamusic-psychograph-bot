# bot.py
import os
import re
import asyncio
from typing import List, Tuple, Optional, Dict

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

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

# SOCKS-прокси для Яндекс Музыки
YM_PROXY_HTTP = os.getenv("YM_PROXY_HTTP", "").strip()   # напр. socks5h://127.0.0.1:9050
YM_PROXY_HTTPS = os.getenv("YM_PROXY_HTTPS", "").strip() # напр. socks5h://127.0.0.1:9050

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


def extract_ym_playlist_urls(text: str) -> List[Tuple[str, int]]:
    """
    Ищет в тексте все ссылки Я.Музыки вида /users/<user>/playlists/<kind>.
    Возвращает пары (user, kind) в порядке появления.
    """
    out: List[Tuple[str, int]] = []
    for m in re.finditer(r"music\.yandex\.(?:ru|com)/users/([^/]+)/playlists/(\d+)", text):
        out.append((m.group(1), int(m.group(2))))
    return out


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
    чтобы любой внутренний requests шёл через SOCKS-туннель.
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
                        "Проверьте, что SOCKS-туннель доступен и YM_PROXY_HTTP/YM_PROXY_HTTPS настроены корректно."
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


def normalize_track_part(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[^0-9a-zа-я]+", " ", s)
    return " ".join(s.split())


def track_key(track: Tuple[str, str]) -> str:
    artist, title = track
    return f"{normalize_track_part(artist)}::{normalize_track_part(title)}"


def get_common_tracks(tracks_a: List[Tuple[str, str]], tracks_b: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    map_a: Dict[str, Tuple[str, str]] = {}
    map_b: Dict[str, Tuple[str, str]] = {}
    for t in tracks_a:
        map_a.setdefault(track_key(t), t)
    for t in tracks_b:
        map_b.setdefault(track_key(t), t)
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    return [map_a[k] for k in keys]


def overlap_stats(tracks_a: List[Tuple[str, str]], tracks_b: List[Tuple[str, str]]) -> Tuple[int, int, float]:
    keys_a = {track_key(t) for t in tracks_a}
    keys_b = {track_key(t) for t in tracks_b}
    overlap = len(keys_a & keys_b)
    union = len(keys_a | keys_b)
    jaccard = overlap / union if union else 0.0
    return overlap, union, jaccard


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


def build_compatibility_messages(
    profile_a: str,
    profile_b: str,
    total_a: int,
    total_b: int,
    overlap: int,
    jaccard: float,
    common_tracks: List[Tuple[str, str]],
) -> List[dict]:
    common_preview = tracks_to_bullets(common_tracks[:50]) if common_tracks else "Нет пересечений по трекам."
    content = (
        "Ниже даны данные по двум людям и их плейлистам.\n"
        "Сделай сводку совместимости по музыкальному вкусу и психологическому портрету.\n\n"
        f"Плейлист A: {total_a} трек(ов)\n"
        f"Плейлист B: {total_b} трек(ов)\n"
        f"Общие треки (уникальные): {overlap}\n"
        f"Индекс пересечения Jaccard: {jaccard:.3f}\n\n"
        "Общие треки (сэмпл):\n"
        f"{common_preview}\n\n"
        "Психографический отчёт A:\n"
        f"{profile_a}\n\n"
        "Психографический отчёт B:\n"
        f"{profile_b}\n\n"
        "Формат ответа:\n"
        "1) Оценка совместимости по вкусам (0-100) с кратким выводом\n"
        "2) Что вас объединяет в музыке\n"
        "3) Возможные различия и точки трения\n"
        "4) Совместимый психологический портрет пары на базе двух профилей\n"
        "5) Практические рекомендации: как слушать музыку вместе, чтобы это усиливало контакт\n"
        "Пиши по-русски. Не используй спецсимволы разметки (#, ##, **, *)."
    )
    return [
        {"role": "system",
         "content": ("Ты музыкальный психолог и аналитик совместимости. "
                     "Не преувеличивай выводы, помечай неопределённость, если данных мало.")},
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


def analyze_compatibility(tracks_a: List[Tuple[str, str]], tracks_b: List[Tuple[str, str]]) -> str:
    overlap, union, jaccard = overlap_stats(tracks_a, tracks_b)
    common_tracks = get_common_tracks(tracks_a, tracks_b)

    if not OPENAI_API_KEY:
        score = int(jaccard * 100)
        preview = tracks_to_bullets(common_tracks[:20]) if common_tracks else "Общие треки не найдены."
        return (
            "Сводка совместимости без LLM (нет OPENAI_API_KEY).\n"
            f"Совпадение по трекам: {overlap} из {union} уникальных, ориентировочно {score}/100.\n\n"
            f"{preview}"
        )

    profile_a = analyze_full_playlist_all_tracks(tracks_a)
    profile_b = analyze_full_playlist_all_tracks(tracks_b)
    return chat_complete(
        build_compatibility_messages(
            profile_a=profile_a,
            profile_b=profile_b,
            total_a=len(tracks_a),
            total_b=len(tracks_b),
            overlap=overlap,
            jaccard=jaccard,
            common_tracks=common_tracks,
        ),
        temperature=0.6,
        max_tokens=1300,
    )


# ==============================
# Telegram bot (Aiogram 3.12)
# ==============================

class InputFlow(StatesGroup):
    awaiting_single_playlist = State()
    awaiting_first_compare_playlist = State()
    awaiting_second_compare_playlist = State()


bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=None),  # по умолчанию без Markdown, чтобы не ловить parse errors
)
dp = Dispatcher()

# Подключаем роутеры
dp.include_router(help_router)
dp.include_router(ads_router)

START_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Мой музыкальный портрет", callback_data="mode:self")],
        [InlineKeyboardButton(text="Совместимость с близким", callback_data="mode:pair")],
    ]
)


async def handle_single_playlist(m: Message, user: str, kind: int):
    await m.answer("Получаю треки, подождите...")

    tracks = await asyncio.to_thread(fetch_tracks, user, kind)
    total = len(tracks)

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

    await m.answer("Минуту, запускаю психографический анализ...")
    analysis = await asyncio.to_thread(analyze_full_playlist_all_tracks, tracks)
    log_event(m.from_user.id, "analysis_done")
    await send_long(m, analysis, parse_mode=None)


async def handle_compare_playlists(
    m: Message,
    playlist_a: Tuple[str, int],
    playlist_b: Tuple[str, int],
):
    (user_a, kind_a), (user_b, kind_b) = playlist_a, playlist_b
    await m.answer("Получаю оба плейлиста, подождите...")

    tracks_a = await asyncio.to_thread(fetch_tracks, user_a, kind_a)
    tracks_b = await asyncio.to_thread(fetch_tracks, user_b, kind_b)
    total_a = len(tracks_a)
    total_b = len(tracks_b)
    common_tracks = get_common_tracks(tracks_a, tracks_b)
    overlap, union, jaccard = overlap_stats(tracks_a, tracks_b)

    preview_a = "\n".join(f"{i}. {a} — {t}" for i, (a, t) in enumerate(tracks_a[:PREVIEW_COUNT], 1))
    preview_b = "\n".join(f"{i}. {a} — {t}" for i, (a, t) in enumerate(tracks_b[:PREVIEW_COUNT], 1))
    common_preview = tracks_to_bullets(common_tracks[:20]) if common_tracks else "Общие треки не найдены."

    await send_long(
        m,
        f"Плейлист A ({user_a}/{kind_a}): {total_a} трек(ов)\n{preview_a}\n\n"
        f"Плейлист B ({user_b}/{kind_b}): {total_b} трек(ов)\n{preview_b}\n\n"
        f"Общих треков (уникальных): {overlap} из {union} (Jaccard: {jaccard:.3f})\n"
        f"{common_preview}",
        parse_mode=None,
    )

    await m.answer("Строю сводку совместимости по вкусам и психологическому портрету...")
    compatibility = await asyncio.to_thread(analyze_compatibility, tracks_a, tracks_b)
    log_event(m.from_user.id, "compat_done")
    await send_long(m, compatibility, parse_mode=None)



@dp.message(CommandStart())
async def on_start(m: Message, state: FSMContext):
    await state.clear()
    upsert_user(m.from_user); log_event(m.from_user.id, "start")
    await m.answer(
        "Выберите формат анализа:\n"
        "1) Личный музыкально-психологический портрет\n"
        "2) Сравнение вкусов и совместимости с близким",
        reply_markup=START_KB
    )


@dp.callback_query(F.data == "mode:self")
async def on_mode_self(c: CallbackQuery, state: FSMContext):
    await state.clear()
    await state.set_state(InputFlow.awaiting_single_playlist)
    await c.answer()
    if not c.message:
        return
    await c.message.answer(
        "Отправьте одну публичную ссылку на плейлист Яндекс Музыки:\n"
        "https://music.yandex.ru/users/<user>/playlists/<kind>"
    )


@dp.callback_query(F.data == "mode:pair")
async def on_mode_pair(c: CallbackQuery, state: FSMContext):
    await state.clear()
    await state.set_state(InputFlow.awaiting_first_compare_playlist)
    await c.answer()
    if not c.message:
        return
    await c.message.answer(
        "Отправьте первую ссылку (ваш плейлист).\n"
        "После этого попрошу вторую ссылку (плейлист друга)."
    )


@dp.message(F.text.contains("music.yandex."))
async def on_text(m: Message, state: FSMContext):
    text = (m.text or "").strip()
    playlists = extract_ym_playlist_urls(text)
    if not playlists:
        await m.answer("Не смог распознать ссылку на плейлист. Нужен формат: https://music.yandex.ru/users/<user>/playlists/<kind>")
        return

    if len(playlists) > 2:
        await m.answer("Нашёл больше двух ссылок. Пришлите одну (для анализа) или две (для сравнения).")
        return

    current_state = await state.get_state()

    try:
        if current_state == InputFlow.awaiting_single_playlist.state:
            if len(playlists) != 1:
                await m.answer("Для личного портрета нужна одна ссылка. Пришлите один плейлист.")
                return
            await handle_single_playlist(m, *playlists[0])
            await state.clear()
            return

        if current_state == InputFlow.awaiting_first_compare_playlist.state:
            if len(playlists) == 2:
                await handle_compare_playlists(m, playlists[0], playlists[1])
                await state.clear()
                return
            await state.update_data(first_playlist=playlists[0])
            await state.set_state(InputFlow.awaiting_second_compare_playlist)
            await m.answer(
                "Принято. Теперь отправьте вторую ссылку (плейлист друга):\n"
                "https://music.yandex.ru/users/<user>/playlists/<kind>"
            )
            return

        if current_state == InputFlow.awaiting_second_compare_playlist.state:
            if len(playlists) != 1:
                await m.answer("На этом шаге нужна одна ссылка. Пришлите второй плейлист одним сообщением.")
                return
            data = await state.get_data()
            first_playlist = data.get("first_playlist")
            if not first_playlist:
                await state.set_state(InputFlow.awaiting_first_compare_playlist)
                await m.answer("Не нашёл первую ссылку в контексте. Пришлите её заново.")
                return
            await handle_compare_playlists(m, tuple(first_playlist), playlists[0])
            await state.clear()
            return

        if len(playlists) == 1:
            await handle_single_playlist(m, *playlists[0])
            return
        await handle_compare_playlists(m, playlists[0], playlists[1])

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
