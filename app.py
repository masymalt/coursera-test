from __future__ import annotations

import os
from typing import Iterable

import requests
from flask import Flask, jsonify, render_template, request
from mutagen import File as MutagenFile

WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
ALLOWED_EXTENSIONS = {".mp3"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB


def _first_non_empty(tag_source: dict[str, list[str]], keys: Iterable[str]) -> str | None:
    for key in keys:
        values = tag_source.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return None


def extract_artist_and_title(file_storage) -> tuple[str | None, str | None]:
    file_storage.stream.seek(0)
    audio = MutagenFile(file_storage.stream, easy=True)
    file_storage.stream.seek(0)

    tags: dict[str, list[str]] = getattr(audio, "tags", {}) or {}
    artist = _first_non_empty(tags, ("artist", "albumartist", "performer", "composer"))
    title = _first_non_empty(tags, ("title",))
    return artist, title


def fetch_artist_photos(artist: str, limit: int = 5) -> list[str]:
    if not artist:
        return []

    search_queries = (
        f"{artist} musician portrait",
        f"{artist} singer",
        artist,
    )
    collected_urls: list[str] = []
    seen: set[str] = set()

    for query in search_queries:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 20,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 800,
        }

        try:
            response = requests.get(WIKIMEDIA_API_URL, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        pages = (payload.get("query") or {}).get("pages") or {}
        for page in pages.values():
            image_info = page.get("imageinfo") or []
            if not image_info:
                continue

            image_url = image_info[0].get("thumburl") or image_info[0].get("url")
            if not image_url or image_url in seen:
                continue

            seen.add(image_url)
            collected_urls.append(image_url)

            if len(collected_urls) >= limit:
                return collected_urls[:limit]

    return collected_urls[:limit]


def allowed_mp3(filename: str) -> bool:
    _, extension = os.path.splitext(filename.lower())
    return extension in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    if "audio_file" not in request.files:
        return jsonify({"error": "Файл не передан."}), 400

    file = request.files["audio_file"]
    if not file.filename:
        return jsonify({"error": "Выберите MP3-файл."}), 400

    if not allowed_mp3(file.filename):
        return jsonify({"error": "Поддерживается только формат MP3."}), 400

    artist, title = extract_artist_and_title(file)

    if not artist and not title:
        return jsonify(
            {
                "error": (
                    "Не удалось распознать теги MP3. "
                    "Проверьте, что в файле заполнены поля Artist и Title."
                )
            }
        ), 422

    photos = fetch_artist_photos(artist or "", limit=5)
    return jsonify(
        {
            "artist": artist or "Неизвестный исполнитель",
            "title": title or "Неизвестное название",
            "photos": photos,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
