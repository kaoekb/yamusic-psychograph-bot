# bot.py
import json
import os
import re
import asyncio
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from yandex_music import Client
from yandex_music.exceptions import NotFoundError, YandexMusicError

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

# Сколько максимум треков отправляем в LLM. 0 = без отдельного лимита.
MAX_TRACKS = int(os.getenv("MAX_TRACKS", "0"))

# Чанкинг для анализа: уходит вся совокупность треков
PROMPT_CHUNK_SIZE = int(os.getenv("PROMPT_CHUNK_SIZE", "220"))
PROMPT_MAX_CHUNKS = int(os.getenv("PROMPT_MAX_CHUNKS", "20"))  # страховка бюджета

# Отправка длинного ответа
MAX_TG = 4000  # безопасный размер куска сообщения (чуть меньше 4096)

BIG_FIVE_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

BIG_FIVE_LABELS = {
    "openness": "Открытость новому",
    "conscientiousness": "Добросовестность",
    "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность",
    "neuroticism": "Нейротизм",
}


@dataclass
class AnalysisScope:
    total_tracks: int
    analyzed_tracks: int
    truncated: bool
    strategy: str
    limit_reason: str = ""


@dataclass
class AnalysisResult:
    text: str
    scope: AnalysisScope
    raw_report: Optional[Dict[str, Any]] = None



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
            except NotFoundError as e:
                msg = str(e)
                if "playlist-not-found" in msg:
                    last_err = RuntimeError(
                        "Плейлист не найден. Возможно, ссылка устарела, плейлист удалён или он не публичный."
                    )
                else:
                    last_err = RuntimeError("Плейлист не найден или недоступен.")
                continue
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


def clamp_score(value: Any, default: int = 50) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, score))


def clean_text(value: Any, default: str = "", max_len: int = 800) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "..."
    return text


def unique_text_list(value: Any, limit: int = 6, max_len: int = 140) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    out: List[str] = []
    seen = set()
    for item in raw_items:
        if isinstance(item, dict):
            item = item.get("name") or item.get("label") or item.get("text") or item.get("value")
        text = clean_text(item, max_len=max_len)
        key = text.lower()
        if not text or key in seen:
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= limit:
            break
    return out


def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(raw: str) -> Dict[str, Any]:
    candidates = [strip_code_fences(raw)]
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        candidates.append(match.group(0))

    last_err: Optional[Exception] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(parsed, dict):
            return parsed
        last_err = RuntimeError("LLM returned JSON, but not an object.")

    raise RuntimeError(f"LLM returned invalid JSON: {last_err}")


def analysis_track_limit() -> Tuple[Optional[int], str]:
    limits: List[Tuple[int, str]] = []
    if MAX_TRACKS > 0:
        limits.append((MAX_TRACKS, f"MAX_TRACKS={MAX_TRACKS}"))
    if PROMPT_CHUNK_SIZE > 0 and PROMPT_MAX_CHUNKS > 0:
        chunk_limit = PROMPT_CHUNK_SIZE * PROMPT_MAX_CHUNKS
        limits.append((chunk_limit, f"PROMPT_CHUNK_SIZE*PROMPT_MAX_CHUNKS={chunk_limit}"))

    if not limits:
        return None, ""

    limit = min(value for value, _ in limits)
    reasons = [label for value, label in limits if value == limit]
    return limit, " и ".join(reasons)


def evenly_sample_tracks(tracks: List[Tuple[str, str]], limit: int) -> List[Tuple[str, str]]:
    if limit >= len(tracks):
        return list(tracks)
    if limit <= 0:
        return []
    if limit == 1:
        return [tracks[0]]

    total = len(tracks)
    sampled: List[Tuple[str, str]] = []
    last_idx = -1
    for i in range(limit):
        idx = round(i * (total - 1) / (limit - 1))
        if idx <= last_idx:
            idx = last_idx + 1
        if idx >= total:
            idx = total - 1
        sampled.append(tracks[idx])
        last_idx = idx
    return sampled


def prepare_tracks_for_analysis(tracks: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], AnalysisScope]:
    total = len(tracks)
    limit, reason = analysis_track_limit()

    if not tracks:
        return [], AnalysisScope(total_tracks=0, analyzed_tracks=0, truncated=False, strategy="empty")

    if limit is None or total <= limit:
        return list(tracks), AnalysisScope(
            total_tracks=total,
            analyzed_tracks=total,
            truncated=False,
            strategy="all_tracks",
        )

    sampled = evenly_sample_tracks(tracks, limit)
    return sampled, AnalysisScope(
        total_tracks=total,
        analyzed_tracks=len(sampled),
        truncated=True,
        strategy="even_sample",
        limit_reason=reason,
    )


def format_analysis_scope_notice(scope: AnalysisScope) -> str:
    if scope.total_tracks == 0:
        return "Плейлист пуст."
    if not scope.truncated:
        return f"В психографический анализ уйдут все {scope.analyzed_tracks} треков."
    return (
        f"В психографический анализ уйдет равномерная выборка {scope.analyzed_tracks} из "
        f"{scope.total_tracks} треков по всему плейлисту. Причина: {scope.limit_reason}."
    )


def normalize_trait_report(raw_trait: Any, default_confidence: int = 55) -> Dict[str, Any]:
    if not isinstance(raw_trait, dict):
        raw_trait = {}
    return {
        "score": clamp_score(raw_trait.get("score"), 50),
        "confidence": clamp_score(raw_trait.get("confidence"), default_confidence),
        "evidence": unique_text_list(raw_trait.get("evidence"), limit=3, max_len=160),
    }


def normalize_music_profile(raw_profile: Any) -> Dict[str, List[str]]:
    if not isinstance(raw_profile, dict):
        raw_profile = {}
    return {
        "genres": unique_text_list(raw_profile.get("genres"), limit=6),
        "moods": unique_text_list(raw_profile.get("moods"), limit=6),
        "themes": unique_text_list(raw_profile.get("themes"), limit=6),
        "languages": unique_text_list(raw_profile.get("languages"), limit=6),
    }


def normalize_chunk_report(raw_report: Dict[str, Any]) -> Dict[str, Any]:
    big_five = raw_report.get("big_five") if isinstance(raw_report.get("big_five"), dict) else {}
    return {
        "chunk_summary": clean_text(
            raw_report.get("chunk_summary"),
            "Часть плейлиста дает смешанный, но читаемый набор сигналов.",
            max_len=400,
        ),
        "music_signals": normalize_music_profile(raw_report.get("music_signals")),
        "big_five": {
            trait: normalize_trait_report(big_five.get(trait), default_confidence=55)
            for trait in BIG_FIVE_TRAITS
        },
        "listener_functions": unique_text_list(raw_report.get("listener_functions"), limit=5),
        "uncertainty_notes": unique_text_list(raw_report.get("uncertainty_notes"), limit=4, max_len=180),
    }


def normalize_final_profile(raw_report: Dict[str, Any]) -> Dict[str, Any]:
    overall_confidence = clamp_score(raw_report.get("overall_confidence"), 60)
    big_five = raw_report.get("big_five") if isinstance(raw_report.get("big_five"), dict) else {}
    profile = {
        "summary": clean_text(
            raw_report.get("summary"),
            "Плейлист дает несколько устойчивых музыкально-психологических сигналов, но выводы остаются вероятностными.",
            max_len=600,
        ),
        "overall_confidence": overall_confidence,
        "big_five": {
            trait: normalize_trait_report(big_five.get(trait), default_confidence=overall_confidence)
            for trait in BIG_FIVE_TRAITS
        },
        "music_profile": normalize_music_profile(raw_report.get("music_profile")),
        "emotional_states": unique_text_list(raw_report.get("emotional_states"), limit=6),
        "listener_functions": unique_text_list(raw_report.get("listener_functions"), limit=6),
        "portrait": clean_text(
            raw_report.get("portrait"),
            "Плейлист указывает на набор устойчивых предпочтений, но не заменяет полноценную психологическую диагностику.",
            max_len=1400,
        ),
        "uncertainty_notes": unique_text_list(raw_report.get("uncertainty_notes"), limit=5, max_len=220),
    }
    return profile


def normalize_compatibility_report(raw_report: Dict[str, Any], jaccard: float) -> Dict[str, Any]:
    return {
        "summary": clean_text(
            raw_report.get("summary"),
            "У пары есть заметные точки соприкосновения в том, как они переживают музыку и что в ней ищут.",
            max_len=700,
        ),
        "shared_patterns": unique_text_list(raw_report.get("shared_patterns"), limit=6, max_len=180),
        "differences": unique_text_list(raw_report.get("differences"), limit=6, max_len=180),
        "friction_points": unique_text_list(raw_report.get("friction_points"), limit=5, max_len=180),
        "pair_portrait": clean_text(
            raw_report.get("pair_portrait"),
            "Музыкальная динамика пары выглядит потенциально живой, но зависит от контекста общения.",
            max_len=1200,
        ),
    }


def render_list_line(label: str, items: List[str], fallback: str = "нет устойчивого сигнала") -> str:
    if items:
        return f"{label}: {', '.join(items)}."
    return f"{label}: {fallback}."


def render_scope_lines(scope: AnalysisScope, prefix: str = "") -> List[str]:
    label = f"{prefix}: " if prefix else ""
    lines = [f"- {label}проанализировано {scope.analyzed_tracks} из {scope.total_tracks} треков."]
    if scope.truncated:
        lines.append(f"- {label}использована равномерная выборка по всему плейлисту.")
        lines.append(f"- {label}сработал лимит {scope.limit_reason}.")
    else:
        lines.append(f"- {label}в анализ вошел весь плейлист.")
    return lines


def normalized_label_set(items: List[str]) -> set[str]:
    return {normalize_track_part(item) for item in items if normalize_track_part(item)}


def list_similarity_score(items_a: List[str], items_b: List[str]) -> Optional[int]:
    set_a = normalized_label_set(items_a)
    set_b = normalized_label_set(items_b)
    if not set_a or not set_b:
        return None
    return round(100 * len(set_a & set_b) / len(set_a | set_b))


def shared_list_items(items_a: List[str], items_b: List[str], limit: int = 4) -> List[str]:
    map_a = {normalize_track_part(item): item for item in items_a if normalize_track_part(item)}
    map_b = {normalize_track_part(item): item for item in items_b if normalize_track_part(item)}
    shared_keys = sorted(set(map_a) & set(map_b))
    return [map_a[key] for key in shared_keys[:limit]]


def weighted_average(scores: List[Tuple[Optional[int], float]], default: int = 50) -> int:
    weighted_sum = 0.0
    total_weight = 0.0
    for score, weight in scores:
        if score is None:
            continue
        weighted_sum += score * weight
        total_weight += weight
    if total_weight == 0:
        return default
    return round(weighted_sum / total_weight)


def trait_similarity_details(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> Tuple[int, List[Tuple[str, int]]]:
    similarities: List[Tuple[str, int]] = []
    adjusted: List[int] = []

    for trait in BIG_FIVE_TRAITS:
        left = profile_a["big_five"][trait]
        right = profile_b["big_five"][trait]
        raw_similarity = max(0, 100 - abs(left["score"] - right["score"]))
        confidence_factor = min(left["confidence"], right["confidence"]) / 100
        adjusted_similarity = round(50 * (1 - confidence_factor) + raw_similarity * confidence_factor)
        similarities.append((trait, adjusted_similarity))
        adjusted.append(adjusted_similarity)

    return round(sum(adjusted) / len(adjusted)), sorted(similarities, key=lambda item: item[1], reverse=True)


def describe_trait_score(trait: str, score: int) -> str:
    if trait == "openness":
        if score >= 67:
            return "Похоже, музыка для вас связана с поиском новизны, оттенков и эмоционально-эстетических открытий."
        if score >= 40:
            return "Скорее всего, вы держите баланс между любопытством к новому и тягой к знакомому музыкальному языку."
        return "Похоже, вам ближе узнаваемость, устойчивый стиль и музыка, которая не требует постоянной перестройки восприятия."
    if trait == "conscientiousness":
        if score >= 67:
            return "Вероятно, вам близок собранный, структурный способ проживать музыку, когда важны форма, ритм и внутренний порядок."
        if score >= 40:
            return "Судя по плейлисту, у вас умеренная тяга к структуре: есть место и порядку, и спонтанности."
        return "Похоже, музыка у вас чаще работает как пространство свободы, импульса и ухода от внутренней регуляции."
    if trait == "extraversion":
        if score >= 67:
            return "Музыка, вероятно, часто нужна вам как источник энергии, включенности и внешнего эмоционального движения."
        if score >= 40:
            return "Похоже, вы соединяете в музыке и внутреннее проживание, и потребность в драйве или контакте."
        return "Скорее всего, музыка для вас чаще про внутренний мир, концентрацию и личное эмоциональное пространство."
    if trait == "agreeableness":
        if score >= 67:
            return "В плейлисте чувствуется тяготение к мягкому, эмпатичному и эмоционально бережному тону."
        if score >= 40:
            return "По музыке видно сочетание теплоты и избирательности: вам важна и близость, и личная граница."
        return "Похоже, музыка у вас нередко про автономию, прямоту или более жесткое эмоциональное высказывание."
    if score >= 67:
        return "Музыка, вероятно, играет заметную роль в проживании напряжения, тревоги и эмоциональной переработке состояний."
    if score >= 40:
        return "Судя по плейлисту, музыка для вас может быть и способом эмоциональной регуляции, и просто источником интереса или удовольствия."
    return "Похоже, музыка у вас чаще связана не с разрядкой тревоги, а с интересом, драйвом, настроением или эстетикой."


def build_trait_block(trait: str, data: Dict[str, Any]) -> List[str]:
    lines = [f"{BIG_FIVE_LABELS[trait]}: {data['score']}/100."]
    lines.append(describe_trait_score(trait, data["score"]))
    if data["evidence"]:
        lines.append(f"Что на это указывает: {'; '.join(data['evidence'])}.")
    return lines


def compute_compatibility_metrics(
    tracks_a: List[Tuple[str, str]],
    tracks_b: List[Tuple[str, str]],
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
    overlap: int,
    union: int,
    jaccard: float,
) -> Dict[str, Any]:
    unique_a = len({track_key(t) for t in tracks_a})
    unique_b = len({track_key(t) for t in tracks_b})
    smaller_ratio = overlap / min(unique_a, unique_b) if min(unique_a, unique_b) else 0.0
    overlap_score = round((0.55 * smaller_ratio + 0.45 * jaccard) * 100)

    genres_score = list_similarity_score(profile_a["music_profile"]["genres"], profile_b["music_profile"]["genres"])
    moods_score = list_similarity_score(profile_a["music_profile"]["moods"], profile_b["music_profile"]["moods"])
    themes_score = list_similarity_score(profile_a["music_profile"]["themes"], profile_b["music_profile"]["themes"])
    languages_score = list_similarity_score(profile_a["music_profile"]["languages"], profile_b["music_profile"]["languages"])
    music_similarity = weighted_average(
        [
            (genres_score, 0.4),
            (moods_score, 0.3),
            (themes_score, 0.2),
            (languages_score, 0.1),
        ],
        default=50,
    )

    emotional_similarity = list_similarity_score(profile_a["emotional_states"], profile_b["emotional_states"])
    function_similarity = list_similarity_score(profile_a["listener_functions"], profile_b["listener_functions"])
    big_five_similarity, trait_pairs = trait_similarity_details(profile_a, profile_b)

    taste_score = round(0.75 * music_similarity + 0.25 * overlap_score)
    psychological_score = weighted_average(
        [
            (big_five_similarity, 0.7),
            (function_similarity, 0.15),
            (emotional_similarity, 0.15),
        ],
        default=55,
    )
    overall_score = round(0.6 * taste_score + 0.4 * psychological_score)

    distant_traits = [BIG_FIVE_LABELS[trait] for trait, value in trait_pairs[-2:] if value < 85]

    return {
        "overall_score": overall_score,
        "taste_score": taste_score,
        "psychological_score": psychological_score,
        "overlap_score": overlap_score,
        "music_similarity": music_similarity,
        "big_five_similarity": big_five_similarity,
        "emotional_similarity": emotional_similarity or 50,
        "function_similarity": function_similarity or 50,
        "shared_genres": shared_list_items(profile_a["music_profile"]["genres"], profile_b["music_profile"]["genres"]),
        "shared_moods": shared_list_items(profile_a["music_profile"]["moods"], profile_b["music_profile"]["moods"]),
        "shared_themes": shared_list_items(profile_a["music_profile"]["themes"], profile_b["music_profile"]["themes"]),
        "shared_functions": shared_list_items(profile_a["listener_functions"], profile_b["listener_functions"]),
        "closest_traits": [BIG_FIVE_LABELS[trait] for trait, _ in trait_pairs[:2]],
        "distant_traits": distant_traits,
        "overlap": overlap,
        "union": union,
        "jaccard": jaccard,
        "unique_a": unique_a,
        "unique_b": unique_b,
    }


def build_compatibility_score_lines(metrics: Dict[str, Any]) -> List[str]:
    shared_music_bits: List[str] = []
    if metrics["shared_genres"]:
        shared_music_bits.append(f"по жанрам вы пересекаетесь в {', '.join(metrics['shared_genres'])}")
    if metrics["shared_moods"]:
        shared_music_bits.append(f"по настроению совпадаете в {', '.join(metrics['shared_moods'])}")
    if metrics["shared_themes"]:
        shared_music_bits.append(f"по темам тянетесь к {', '.join(metrics['shared_themes'])}")

    music_reason = "; ".join(shared_music_bits) if shared_music_bits else "главное совпадение видно не в конкретных треках, а в общем музыкальном языке"
    psych_reason_bits: List[str] = []
    if metrics["closest_traits"]:
        psych_reason_bits.append(f"ближе всего вы по шкалам {', '.join(metrics['closest_traits'])}")
    if metrics["shared_functions"]:
        psych_reason_bits.append(f"музыку вы используете похоже: {', '.join(metrics['shared_functions'])}")
    psych_reason = "; ".join(psych_reason_bits) if psych_reason_bits else "в музыке вы похожи по способу эмоциональной настройки"

    return [
        f"- Общая совместимость: {metrics['overall_score']}/100.",
        "  Почему: у вас заметно совпадает не только набор музыкальных симпатий, но и способ эмоционально пользоваться музыкой.",
        f"- Совпадение по вкусам: {metrics['taste_score']}/100.",
        f"  Почему: {music_reason}; пересечение по конкретным трекам — {metrics['overlap']} из {metrics['union']} уникальных.",
        f"- Психологическая сочетаемость: {metrics['psychological_score']}/100.",
        f"  Почему: {psych_reason}.",
    ]


def format_playlist_analysis(profile: Dict[str, Any], scope: AnalysisScope) -> str:
    lines: List[str] = [
        "Музыкально-психологический портрет",
        profile["summary"],
        "",
        "Big Five",
    ]

    for i, trait in enumerate(BIG_FIVE_TRAITS, 1):
        data = profile["big_five"][trait]
        lines.append(f"{i}. {BIG_FIVE_LABELS[trait]}")
        lines.extend(build_trait_block(trait, data))
        lines.append("")

    lines.extend([
        "Музыкальный язык",
        render_list_line("- Жанры", profile["music_profile"]["genres"]),
        render_list_line("- Настроения", profile["music_profile"]["moods"]),
        render_list_line("- Темы", profile["music_profile"]["themes"]),
        render_list_line("- Языки", profile["music_profile"]["languages"]),
        "",
        "Эмоциональный рисунок",
        render_list_line("- Эмоциональные состояния", profile["emotional_states"]),
        "",
        "Зачем вам нужен этот плейлист",
        render_list_line("- Функции плейлиста", profile["listener_functions"]),
        "",
        "Развернутый портрет",
        profile["portrait"],
    ])

    return "\n".join(lines)


def format_compatibility_analysis(
    report: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    lines: List[str] = [
        "Краткий вывод",
        report["summary"],
        "",
        "Оценки",
        *build_compatibility_score_lines(metrics),
    ]

    if report["shared_patterns"]:
        lines.extend(["", "Что вас объединяет"])
        lines.extend(f"- {item}" for item in report["shared_patterns"])

    if report["differences"]:
        lines.extend(["", "Различия"])
        lines.extend(f"- {item}" for item in report["differences"])

    if report["friction_points"]:
        lines.extend(["", "Точки трения"])
        lines.extend(f"- {item}" for item in report["friction_points"])

    lines.extend(["", "Совместный портрет", report["pair_portrait"]])

    return "\n".join(lines)


def build_chunk_messages(k: int, n: int, chunk: List[Tuple[str, str]]) -> List[dict]:
    bullets = tracks_to_bullets(chunk)
    content = (
        f"ЧАСТЬ {k}/{n}. Ниже список из {len(chunk)} треков.\n\n"
        f"{bullets}\n\n"
        "Верни только валидный JSON-объект без markdown и без пояснений вне JSON.\n"
        "Не делай финальных выводов по всему плейлисту. Анализируй только эту часть.\n"
        "Если сигнал слабый, снижай confidence. Не выдумывай недоступные факты.\n"
        "Схема JSON:\n"
        "{\n"
        '  "chunk_summary": "1-2 предложения",\n'
        '  "music_signals": {\n'
        '    "genres": ["до 5 пунктов"],\n'
        '    "moods": ["до 5 пунктов"],\n'
        '    "themes": ["до 5 пунктов"],\n'
        '    "languages": ["до 5 пунктов"]\n'
        "  },\n"
        '  "big_five": {\n'
        '    "openness": {"score": 0, "confidence": 0, "evidence": ["до 3 коротких сигналов"]},\n'
        '    "conscientiousness": {"score": 0, "confidence": 0, "evidence": ["до 3 коротких сигналов"]},\n'
        '    "extraversion": {"score": 0, "confidence": 0, "evidence": ["до 3 коротких сигналов"]},\n'
        '    "agreeableness": {"score": 0, "confidence": 0, "evidence": ["до 3 коротких сигналов"]},\n'
        '    "neuroticism": {"score": 0, "confidence": 0, "evidence": ["до 3 коротких сигналов"]}\n'
        "  },\n"
        '  "listener_functions": ["до 5 пунктов"],\n'
        '  "uncertainty_notes": ["до 4 пунктов"]\n'
        "}\n"
        "Все score/confidence - целые числа от 0 до 100."
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты специалист по музыкальной психографии в парадигме Big Five. "
                "Работай осторожно, опирайся на повторяющиеся паттерны жанров, настроений, тем и языков. "
                "Возвращай только JSON."
            ),
        },
        {"role": "user", "content": content},
    ]


def build_synthesis_messages(partials: List[Dict[str, Any]], scope: AnalysisScope) -> List[dict]:
    coverage_note = (
        f"В анализ ушли все {scope.analyzed_tracks} треков."
        if not scope.truncated
        else (
            f"В анализ ушла равномерная выборка {scope.analyzed_tracks} из {scope.total_tracks} треков "
            f"по всему плейлисту. Причина: {scope.limit_reason}."
        )
    )
    partials_json = json.dumps(partials, ensure_ascii=False, separators=(",", ":"))
    content = (
        "Ниже JSON-отчеты по частям одного плейлиста.\n"
        f"{coverage_note}\n\n"
        "Объедини их в один финальный JSON-объект без markdown.\n"
        "Старайся делать выводы только там, где признаки повторяются между частями. "
        "Если данные неоднородны, это должно снижать overall_confidence.\n"
        "Схема JSON:\n"
        "{\n"
        '  "summary": "3-5 предложений",\n'
        '  "overall_confidence": 0,\n'
        '  "big_five": {\n'
        '    "openness": {"score": 0, "confidence": 0, "evidence": ["до 3 пунктов"]},\n'
        '    "conscientiousness": {"score": 0, "confidence": 0, "evidence": ["до 3 пунктов"]},\n'
        '    "extraversion": {"score": 0, "confidence": 0, "evidence": ["до 3 пунктов"]},\n'
        '    "agreeableness": {"score": 0, "confidence": 0, "evidence": ["до 3 пунктов"]},\n'
        '    "neuroticism": {"score": 0, "confidence": 0, "evidence": ["до 3 пунктов"]}\n'
        "  },\n"
        '  "music_profile": {\n'
        '    "genres": ["до 6 пунктов"],\n'
        '    "moods": ["до 6 пунктов"],\n'
        '    "themes": ["до 6 пунктов"],\n'
        '    "languages": ["до 6 пунктов"]\n'
        "  },\n"
        '  "emotional_states": ["до 6 пунктов"],\n'
        '  "listener_functions": ["до 6 пунктов"],\n'
        '  "portrait": "6-10 предложений",\n'
        '  "uncertainty_notes": ["до 5 пунктов"]\n'
        "}\n\n"
        f"Исходные части:\n{partials_json}"
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты специалист по музыкальной психографии. "
                "Собери стабильный итоговый профиль только на основе присланных структурированных chunk-отчетов. "
                "Возвращай только JSON."
            ),
        },
        {"role": "user", "content": content},
    ]


def build_compatibility_messages(
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
    metrics: Dict[str, Any],
) -> List[dict]:
    payload = {
        "playlist_a": {
            "profile": profile_a,
        },
        "playlist_b": {
            "profile": profile_b,
        },
        "metrics": metrics,
    }
    content = (
        "Ниже структурированные профили двух людей и уже посчитанные метрики совместимости.\n"
        "Сделай психологически аккуратное описание совместимости.\n"
        "Верни только валидный JSON-объект без markdown.\n"
        "Важно: не своди совместимость только к общим трекам. "
        "Если музыкальный язык, настроение, темы и способ использовать музыку совпадают, это тоже признак близости.\n"
        "Схема JSON:\n"
        "{\n"
        '  "summary": "3-5 предложений",\n'
        '  "shared_patterns": ["до 6 пунктов"],\n'
        '  "differences": ["до 6 пунктов"],\n'
        '  "friction_points": ["до 5 пунктов"],\n'
        '  "pair_portrait": "5-8 предложений"\n'
        "}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты музыкальный психолог и аналитик совместимости. "
                "Работай аккуратно и возвращай только JSON."
            ),
        },
        {"role": "user", "content": content},
    ]


def openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def pick_model_sequence() -> List[str]:
    seq: List[str] = []
    if MODEL:
        seq.append(MODEL)
    seq += ["gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"]
    seen, out = set(), []
    for model in seq:
        if model and model not in seen:
            out.append(model)
            seen.add(model)
    return out


def chat_complete(
    messages: List[dict],
    temperature: float = 0.7,
    max_tokens: int = 900,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    cli = openai_client()
    last_err = None
    for model in pick_model_sequence():
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            r = cli.chat.completions.create(**kwargs)
            content = (r.choices[0].message.content or "").strip()
            if content:
                return content
            last_err = RuntimeError(f"Model {model} returned an empty response.")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"OpenAI недоступен или модель запрещена: {last_err}")


def chat_complete_json(messages: List[dict], temperature: float = 0.3, max_tokens: int = 1200) -> Dict[str, Any]:
    try:
        raw = chat_complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return parse_json_object(raw)
    except Exception:
        raw = chat_complete(messages, temperature=temperature, max_tokens=max_tokens)
        return parse_json_object(raw)


def analyze_full_playlist_all_tracks(tracks: List[Tuple[str, str]]) -> AnalysisResult:
    """
    Полный анализ плейлиста через чанки + структурированную финальную синтезу.
    """
    scoped_tracks, scope = prepare_tracks_for_analysis(tracks)

    if not OPENAI_API_KEY:
        return AnalysisResult(
            text=(
                "Аналитика отключена: нет OPENAI_API_KEY.\n"
                "Треки получены, но для психографического анализа нужен ключ OpenAI."
            ),
            scope=scope,
        )

    if not tracks:
        return AnalysisResult(text="Плейлист пуст.", scope=scope)

    chunks: List[List[Tuple[str, str]]] = [
        scoped_tracks[i:i + PROMPT_CHUNK_SIZE] for i in range(0, len(scoped_tracks), PROMPT_CHUNK_SIZE)
    ]

    partials: List[Dict[str, Any]] = []
    n = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        chunk_json = chat_complete_json(build_chunk_messages(i, n, chunk), temperature=0.25, max_tokens=1100)
        partials.append(normalize_chunk_report(chunk_json))

    final_json = chat_complete_json(build_synthesis_messages(partials, scope), temperature=0.2, max_tokens=1800)
    final_profile = normalize_final_profile(final_json)
    return AnalysisResult(
        text=format_playlist_analysis(final_profile, scope),
        scope=scope,
        raw_report=final_profile,
    )


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

    result_a = analyze_full_playlist_all_tracks(tracks_a)
    result_b = analyze_full_playlist_all_tracks(tracks_b)
    profile_a = result_a.raw_report or {}
    profile_b = result_b.raw_report or {}
    metrics = compute_compatibility_metrics(
        tracks_a=tracks_a,
        tracks_b=tracks_b,
        profile_a=profile_a,
        profile_b=profile_b,
        overlap=overlap,
        union=union,
        jaccard=jaccard,
    )

    compatibility_json = chat_complete_json(
        build_compatibility_messages(
            profile_a=profile_a,
            profile_b=profile_b,
            metrics=metrics,
        ),
        temperature=0.25,
        max_tokens=1500,
    )
    compatibility = normalize_compatibility_report(compatibility_json, jaccard=jaccard)
    return format_compatibility_analysis(compatibility, metrics)


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
    _, scope = prepare_tracks_for_analysis(tracks)

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

    if scope.truncated:
        await m.answer(format_analysis_scope_notice(scope), parse_mode=None)

    await m.answer("Минуту, запускаю психографический анализ...")
    analysis = await asyncio.to_thread(analyze_full_playlist_all_tracks, tracks)
    log_event(m.from_user.id, "analysis_done")
    await send_long(m, analysis.text, parse_mode=None)


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
    _, scope_a = prepare_tracks_for_analysis(tracks_a)
    _, scope_b = prepare_tracks_for_analysis(tracks_b)

    await send_long(
        m,
        "Общие треки\n"
        f"Найдено {overlap} общих треков.\n\n"
        f"{tracks_to_bullets(common_tracks[:50]) if common_tracks else 'Общие треки не найдены.'}",
        parse_mode=None,
    )

    truncation_notes: List[str] = []
    if scope_a.truncated:
        truncation_notes.append(f"Для плейлиста A: {format_analysis_scope_notice(scope_a)}")
    if scope_b.truncated:
        truncation_notes.append(f"Для плейлиста B: {format_analysis_scope_notice(scope_b)}")
    if truncation_notes:
        await send_long(m, "\n".join(truncation_notes), parse_mode=None)

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
