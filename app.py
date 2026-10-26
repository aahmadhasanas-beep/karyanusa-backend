"""Backend Karya Nusa - perantara aman antara website dan Gemini API.

API key HANYA dibaca dari environment variable GEMINI_API_KEY.
Tidak pernah dikirim ke browser dan tidak ditulis di file ini.
"""
import json
import os
import re
import time
from collections import defaultdict, deque

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024  # batas ukuran permintaan: 32 KB

# --- Konfigurasi (semua dari environment variable) -------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://aahmadhasanas-beep.github.io"
    ).split(",")
    if o.strip()
]
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "12"))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}}, methods=["POST", "OPTIONS"])

# --- Pembatas permintaan sederhana (per IP, di memori) ---------------------
_hits = defaultdict(deque)


def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "?")


@app.before_request
def rate_limit():
    if not request.path.startswith("/api/") or request.method == "OPTIONS":
        return None
    now = time.time()
    q = _hits[_client_ip()]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        return err("Terlalu banyak permintaan. Tunggu sekitar satu menit lalu coba lagi.", 429)
    q.append(now)
    return None


def err(message, status=400):
    return jsonify({"ok": False, "error": message}), status


# --- Pemanggil Gemini ------------------------------------------------------
class GeminiError(Exception):
    pass


def call_gemini(prompt, system, json_mode=False, max_tokens=8192, use_search=False):
    if not GEMINI_API_KEY:
        raise GeminiError("Server belum dikonfigurasi (GEMINI_API_KEY belum diatur).")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_tokens},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if use_search:
        body["tools"] = [{"google_search": {}}]
    try:
        r = requests.post(
            GEMINI_URL.format(model=GEMINI_MODEL),
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=90,
        )
    except requests.Timeout:
        raise GeminiError("Layanan AI terlalu lama merespons. Silakan coba lagi.")
    except requests.RequestException:
        raise GeminiError("Tidak dapat menghubungi layanan AI. Silakan coba lagi.")

    if r.status_code == 429:
        raise GeminiError("Kuota AI gratis sedang penuh. Tunggu beberapa menit lalu coba lagi.")
    if r.status_code in (400, 403) :
        app.logger.error("Gemini %s: %s", r.status_code, r.text[:300])
        raise GeminiError("Permintaan ditolak layanan AI (periksa API key atau nama model di server).")
    if r.status_code == 404:
        raise GeminiError("Model AI tidak ditemukan. Periksa nilai GEMINI_MODEL di server.")
    if r.status_code >= 500:
        raise GeminiError("Layanan AI sedang bermasalah. Silakan coba lagi nanti.")
    if r.status_code != 200:
        app.logger.error("Gemini %s: %s", r.status_code, r.text[:300])
        raise GeminiError("Terjadi kesalahan pada layanan AI.")

    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise GeminiError("AI tidak mengembalikan hasil (kemungkinan diblokir filter keamanan). Ubah prompt Anda.")
    cand = cands[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
    if not text:
        raise GeminiError("AI mengembalikan hasil kosong. Silakan coba lagi.")
    sources = []
    for ch in (cand.get("groundingMetadata", {}) or {}).get("groundingChunks", []) or []:
        web = ch.get("web") or {}
        if web.get("uri"):
            sources.append({"title": web.get("title", ""), "url": web["uri"]})
    return text, sources, cand.get("finishReason", "")


SYSTEM = (
    "Anda penulis akademik berbahasa Indonesia. Tulis dengan bahasa akademik yang jelas, "
    "mengalir, dan mendalam. DILARANG mengarang referensi, nama penulis, tahun, DOI, URL, "
    "nomor halaman, atau kutipan langsung. Jika Anda tidak yakin suatu fakta, tulis secara "
    "umum atau beri tanda [PERLU DIVERIFIKASI]. Jangan mengulang paragraf untuk memenuhi panjang."
)


# --- Validasi --------------------------------------------------------------
def get_json():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Format permintaan tidak valid.")
    return data


def clean_text(value, name, max_len, required=True):
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    value = value.strip()
    if required and not value:
        raise ValueError(f"{name} wajib diisi.")
    if len(value) > max_len:
        raise ValueError(f"{name} terlalu panjang (maksimal {max_len} karakter).")
    return value


def clean_int(value, name, lo, hi):
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} harus berupa angka.")
    if not lo <= n <= hi:
        raise ValueError(f"{name} harus antara {lo} dan {hi}.")
    return n


# Perkiraan kasar: A4, TNR 12, spasi 1,5 ~ 330 kata per halaman isi.
WORDS_PER_PAGE = 330


# --- Endpoint --------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "karyanusa-backend", "ai_configured": bool(GEMINI_API_KEY)})


@app.post("/api/outline")
def outline():
    """Langkah 1: AI menyusun kerangka makalah (JSON) sesuai prompt dan jumlah halaman."""
    try:
        d = get_json()
        prompt = clean_text(d.get("prompt"), "Prompt", 4000)
        pages = clean_int(d.get("pages"), "Jumlah halaman", 3, 60)
        structure = clean_text(d.get("structure", ""), "Struktur khusus", 1500, required=False)
    except ValueError as e:
        return err(str(e))

    # Halaman isi = total dikurangi sampul, kata pengantar, daftar isi, daftar pustaka (perkiraan)
    body_pages = max(2, pages - 4)
    total_words = body_pages * WORDS_PER_PAGE
    ask = f"""Permintaan pengguna:
\"\"\"{prompt}\"\"\"

Target makalah: sekitar {pages} halaman (isi pokok sekitar {total_words} kata).
Struktur khusus dari pengguna (jika ada): {structure or "tidak ada, gunakan struktur standar BAB I Pendahuluan, BAB II Pembahasan, BAB III Penutup"}

Susun kerangka makalah. Jawab HANYA JSON dengan bentuk:
{{"title": "judul makalah", "sections": [{{"heading": "BAB I PENDAHULUAN / subbab", "points": ["poin yang harus dibahas", "..."], "words": 400}}]}}
Aturan: total "words" semua section sekitar {total_words}; setiap bagian cukup kecil (maksimal 900 kata) agar bisa ditulis satu per satu; sesuaikan bagian dengan topik, bukan generik; jangan sertakan sampul, daftar isi, atau daftar pustaka (itu dibuat terpisah)."""
    try:
        text, _, _ = call_gemini(ask, SYSTEM, json_mode=True, max_tokens=4096)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
        secs = data["sections"]
        assert isinstance(secs, list) and secs
        result = {
            "title": str(data.get("title", ""))[:300],
            "sections": [
                {
                    "heading": str(s["heading"])[:200],
                    "points": [str(p)[:300] for p in s.get("points", [])][:12],
                    "words": max(100, min(900, int(s.get("words", 300)))),
                }
                for s in secs[:40]
            ],
        }
    except GeminiError as e:
        return err(str(e), 502)
    except (ValueError, KeyError, TypeError, AssertionError):
        return err("AI mengembalikan kerangka dengan format yang tidak terbaca. Silakan coba lagi.", 502)
    return jsonify({"ok": True, **result})


@app.post("/api/section")
def section():
    """Langkah 2 (diulang per bagian): AI menulis satu bagian makalah."""
    try:
        d = get_json()
        prompt = clean_text(d.get("prompt"), "Prompt", 4000)
        title = clean_text(d.get("title"), "Judul", 300)
        heading = clean_text(d.get("heading"), "Judul bagian", 200)
        words = clean_int(d.get("words"), "Jumlah kata", 100, 900)
        points = d.get("points", [])
        if not isinstance(points, list) or len(points) > 12:
            raise ValueError("Daftar poin tidak valid.")
        points = [clean_text(p, "Poin", 300) for p in points]
        outline = d.get("outline", [])
        if not isinstance(outline, list) or len(outline) > 40:
            raise ValueError("Kerangka tidak valid.")
        outline = [clean_text(h, "Kerangka", 200) for h in outline]
        previous = clean_text(d.get("previous", ""), "Ringkasan sebelumnya", 3000, required=False)
        use_search = bool(d.get("use_search", False))
    except ValueError as e:
        return err(str(e))

    ask = f"""Judul makalah: {title}
Permintaan asli pengguna: \"\"\"{prompt}\"\"\"
Kerangka lengkap: {" | ".join(outline)}
Akhir bagian sebelumnya (agar nyambung, jangan diulang): {previous or "(awal makalah)"}

Tulis HANYA bagian: "{heading}"
Poin yang harus dibahas: {"; ".join(points) or "sesuaikan dengan topik"}
Panjang sekitar {words} kata. Tulis paragraf utuh berbahasa akademik. Boleh memakai subjudul
dengan awalan "## ". Jangan menulis judul bagian itu sendiri di awal, jangan memakai poin
berlebihan, jangan membuat daftar pustaka, dan jangan mengarang kutipan atau sumber."""
    try:
        text, sources, finish = call_gemini(
            ask, SYSTEM, max_tokens=4096, use_search=use_search
        )
    except GeminiError as e:
        return err(str(e), 502)
    return jsonify({"ok": True, "heading": heading, "text": text, "sources": sources,
                    "truncated": finish == "MAX_TOKENS"})


@app.errorhandler(413)
def too_large(_):
    return err("Permintaan terlalu besar.", 413)


@app.errorhandler(404)
def not_found(_):
    return err("Alamat tidak ditemukan.", 404)


@app.errorhandler(405)
def bad_method(_):
    return err("Metode tidak diizinkan.", 405)


@app.errorhandler(500)
def server_error(_):
    return err("Terjadi kesalahan di server.", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
