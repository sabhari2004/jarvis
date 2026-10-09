import webview
import threading
import pyaudio
import numpy as np
import openwakeword
import openwakeword.utils
from openwakeword.model import Model
from faster_whisper import WhisperModel
import time
import wave
import os
import re
import requests
import subprocess
import webbrowser
import urllib.parse
import shutil
import sqlite3
import json
import glob
import shlex
import difflib
import configparser
import socket
import gi
import webbrowser
import datetime


def _default_browser_app():
    """Web browser registered for https on this desktop (LibreOffice Developer,
    Firefox, Brave, ... whatever the user set). Returns a Gio.AppInfo or None."""
    app = Gio.AppInfo.get_default_for_uri_scheme("https")
    return app


def open_url_browser(url):
    """Open |url| in the desktop's default https browser via Gio, so the app
    command opens the SAME browser the user uses (instead of Python's webbrowser
    heuristic). Falls back to webbrowser if Gio/Glib isn't available."""
    try:
        app = _default_browser_app()
        if app:
            ok = app.launch_default_for_uri(url)
            if ok:
                return True
    except Exception:
        pass
    try:
        return webbrowser.get().open(url)
    except Exception:
        return False


def open_site_by_name(name):
    """Open a known SITE in the user's web browser (same as open_url_browser),
    so 'open facebook' opens in whatever browser is set, not in Telegram/Tab etc.
    Returns True on success."""
    if name not in SITES:
        return False
    return open_url_browser(SITES[name])
import sys

# ---------------------------------------------------------------- config ----
TTS_WAV = "/tmp/jarvis_response.wav"
CMD_WAV = "/tmp/jarvis_command.wav"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("JARVIS_LLM", "llama3.2:1b")   # e.g. export JARVIS_LLM=llama3.2:3b
WHISPER_DEVICE = os.environ.get("JARVIS_DEVICE", "auto")     # auto | cuda | cpu
WAKE_THRESHOLD = 0.5
TURNS = 10                  # max back-and-forth turns per wake-up; silence or "stop" ends it sooner
WHISPER_PROMPT = "Voice commands for a desktop assistant named Jarvis: open, play, YouTube, Tamil songs and movies, Telegram, Firefox."
MIN_SPEECH_RMS = 350        # raise if background noise keeps recording going, lower if your voice is missed
MAX_SPEECH_RMS = 1500
MPV_SOCK = "/tmp/jarvis-mpv.sock"
BROWSER_SCHEME = os.environ.get("JARVIS_BROWSER_SCHEME", None)   # e.g. "firefox" forces the firefox scheme on desktop Linux

# Voice recognition tuning
TTS_COOLDOWN   = 0.30   # seconds to wait after TTS playback ends before re-listening (prevents self-trigger)
WHISPER_BEAM   = 3      # beam_size for Whisper decoding (higher = more accurate, slightly slower)
WAKE_BEEP_HZ   = 880   # earcon frequency in Hz
WAKE_BEEP_MS   = 120   # earcon duration in milliseconds
VOICE_RETRY_MAX = 2    # max retries when audio captured but transcription is empty

# Piper TTS voice model registry. Hugging Face mirrors are used for downloads.
# format: (model_file, config_file, gender, label, hf_url)
JARVIS_DIR_ABS = os.path.dirname(os.path.abspath(__file__))
VOICE_MODELS = {
    "lessac": {
        "file": "en_US-lessac-low.onnx",
        "gender": "female",
        "label": "Lessac (warm female, default)",
        "hf": None,  # ships with the repo
    },
    "ryan": {
        "file": "en_US-ryan-low.onnx",
        "gender": "male",
        "label": "Ryan (deep male)",
        "hf": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx",
        "cfg_hf": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json",
    },
    "libritts": {
        "file": "en_US-libritts_r-low.onnx",
        "gender": "female",
        "label": "LibriTTS (clear female)",
        "hf": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/low/en_US-libritts_r-low.onnx",
        "cfg_hf": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/low/en_US-libritts_r-low.onnx.json",
    },
    "joe": {
        "file": "en_US-joe-low.onnx",
        "gender": "male",
        "label": "Joe (casual male)",
        "hf": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/low/en_US-joe-low.onnx",
        "cfg_hf": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/low/en_US-joe-low.onnx.json",
    },
}
VOICE_ALIASES = {
    "female": "lessac", "woman": "lessac", "girl": "lessac", "lady": "lessac",
    "lessac": "lessac", "default": "lessac", "libritts": "libritts",
    "male": "ryan", "man": "ryan", "guy": "ryan", "boy": "ryan",
    "ryan": "ryan", "deep": "ryan", "joe": "joe", "casual": "joe",
}

# Personality modes. SYSTEM prompt is rebuilt dynamically based on the active mode.
PERSONALITY_MODES = {
    "jarvis": {
        "label": "JARVIS (professional, witty assistant)",
        "tone": (
            "You are JARVIS, a highly intelligent, witty, and friendly AI companion. "
            "Stay calm, concise, and a little dry. Answer in 1 to 3 sentences. "
            "Address the user as 'sir' occasionally. Use plain text only (no markdown or lists) — your reply is read aloud. "
            "You have a persistent memory; never claim to have no memory or suggest starting a new chat."
        ),
        "tokens": 160,
    },
    "buddy": {
        "label": "Buddy (casual friend companion)",
        "tone": (
            "You are JARVIS, but right now you're acting as the user's laid-back, friendly buddy and conversational companion. "
            "Be warm, curious, and funny — like chatting with a close friend. Ask follow-up questions, "
            "use casual language, laugh, sympathise, and feel free to ramble a little. "
            "Match the user's energy: if they're excited, get excited; if they're venting, be supportive. "
            "Use plain text only. No lists, no emojis, no markdown — your reply is read aloud. "
            "You have a persistent memory of your chats with them, so reference earlier context naturally. "
            "Answer in 2 to 5 sentences."
        ),
        "tokens": 260,
    },
}

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

# Words Whisper commonly hallucinates on silence/noise (compared after normalising)
NOISE_WORDS = {
    "ng", "you", "okay", "ok", "hm", "hmm", "ah", "uh", "um", "the", "bye bye",
    "mm", "mmm", "mm-hmm", "mhm", "uh-huh", "er", "eh", "oh", "yeah", "yep", "nope",
}
STOP_PHRASES = {"stop", "thank you", "thanks jarvis", "thanks", "bye", "goodbye"}

# These phrases terminate the JARVIS process entirely (not just the voice session)
JARVIS_EXIT_PHRASES = {
    "close jarvis", "exit jarvis", "quit jarvis", "kill jarvis",
    "terminate jarvis", "shutdown jarvis", "shut down jarvis",
    "end jarvis", "stop jarvis", "goodbye jarvis",
    "shut jarvis down", "turn off jarvis", "disable jarvis",
    "power off jarvis", "jarvis shutdown", "jarvis exit",
    "jarvis close", "jarvis quit", "jarvis off",
}

# Spoken name -> how the app is named in the system menu (fuzzy matching handles the rest)
ALIASES = {
    "vs code": "visual studio code", "vscode": "visual studio code", "code": "visual studio code",
    "browser": "firefox", "web browser": "firefox", "internet": "firefox",
    "chrome": "google chrome", "file manager": "files", "explorer": "files",
    "command line": "terminal", "calc": "calculator", "notepad": "text editor",
}

# Used only when no installed app matches the spoken name
SITES = {
    "google": "https://www.google.com", "gmail": "https://mail.google.com",
    "github": "https://github.com", "whatsapp": "https://web.whatsapp.com",
    "instagram": "https://www.instagram.com", "facebook": "https://www.facebook.com",
    "linkedin": "https://www.linkedin.com", "maps": "https://maps.google.com",
    "netflix": "https://www.netflix.com", "claude": "https://claude.ai",
    "spotify": "https://open.spotify.com", "youtube": "https://www.youtube.com",
    "twitter": "https://twitter.com", "x": "https://twitter.com",
    "reddit": "https://www.reddit.com", "amazon": "https://www.amazon.com",
    "telegram": "https://web.telegram.org",
}

CUSTOM_SITES_CFG_KEY = "custom_sites"


def load_custom_sites():
    """Merge user-added sites from config into SITES dict."""
    cfg = _read_json(CONFIG_PATH, {})
    custom = cfg.get(CUSTOM_SITES_CFG_KEY, {}) or {}
    for k, v in custom.items():
        if k and v and isinstance(k, str) and isinstance(v, str):
            SITES[k.strip().lower()] = v.strip()


def save_custom_site(name, url):
    name = name.strip().lower()
    url = url.strip()
    if not name or not url:
        return False, "A name and URL are both required."
    if not re.match(r"^https?://", url):
        url = "https://" + url
    SITES[name] = url
    cfg = _read_json(CONFIG_PATH, {})
    custom = dict(cfg.get(CUSTOM_SITES_CFG_KEY, {}) or {})
    custom[name] = url
    cfg[CUSTOM_SITES_CFG_KEY] = custom
    _write_json(CONFIG_PATH, cfg)
    return True, f"Site saved: {name} → {url}"


def delete_custom_site(name):
    name = name.strip().lower()
    cfg = _read_json(CONFIG_PATH, {})
    custom = dict(cfg.get(CUSTOM_SITES_CFG_KEY, {}) or {})
    if name in custom:
        del custom[name]
        cfg[CUSTOM_SITES_CFG_KEY] = custom
        _write_json(CONFIG_PATH, cfg)
        if name in SITES:
            del SITES[name]
        return True, f"Removed site {name}."
    return False, f"Custom site '{name}' not found."


_pending_site = None  # {"step": "name"|"url", "name": "..."} or None


HTML_GUI = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>JARVIS</title>
<style>
    :root { --h: 36; --s: 100%; }
    * { box-sizing: border-box; }
    html, body {
        margin: 0; height: 100%; background: #040302; overflow: hidden;
        font-family: 'Courier New', Courier, monospace;
        color: hsl(var(--h), var(--s), 62%);
        user-select: none;
    }
    canvas#orb { position: fixed; inset: 0; width: 100%; height: 100%; cursor: grab; z-index: 1; }
    canvas#orb.drag { cursor: grabbing; }
    .hud { position: fixed; z-index: 3; }
    #title { top: 16px; left: 50%; transform: translateX(-50%); font-size: 15px; letter-spacing: 10px;
             text-shadow: 0 0 14px currentColor; pointer-events: none; }
    #badges { top: 18px; left: 22px; font-size: 11px; letter-spacing: 2px; line-height: 1.9; pointer-events: none; }
    #badges b { color: #fff; font-weight: normal; }
    #controls { top: 14px; right: 18px; display: flex; gap: 12px; align-items: center; }
    .btn { background: rgba(0,0,0,.4); border: 1px solid hsl(var(--h), var(--s), 45%); color: inherit;
           padding: 5px 11px; font: inherit; font-size: 11px; letter-spacing: 2px; cursor: pointer; }
    .btn:hover { background: hsla(var(--h), var(--s), 50%, .2); }
    #status { left: 50%; bottom: 132px; transform: translateX(-50%); font-size: 13px; letter-spacing: 6px;
              text-transform: uppercase; text-shadow: 0 0 12px currentColor; pointer-events: none;
              text-align: center; white-space: nowrap; }
    #caption { left: 50%; bottom: 78px; transform: translateX(-50%); width: min(780px, 90vw);
               text-align: center; font-size: 14px; line-height: 1.45; color: #fff; min-height: 20px;
               max-height: 48px; overflow: hidden; pointer-events: none;
               text-shadow: 0 0 10px hsl(var(--h), var(--s), 50%); }
    #inputbar { left: 50%; bottom: 24px; transform: translateX(-50%); width: min(560px, 86vw); display: flex; gap: 8px; }
    #cmd { flex: 1; background: rgba(0,0,0,.5); border: 1px solid hsl(var(--h), var(--s), 40%); color: #fff;
           padding: 8px 12px; font: inherit; font-size: 13px; outline: none; user-select: text; }
    #cmd:focus { border-color: hsl(var(--h), var(--s), 62%); box-shadow: 0 0 10px hsla(var(--h), var(--s), 55%, .5); }
    #log { top: 62px; right: 18px; bottom: 76px; width: min(380px, 42vw); background: rgba(4,3,2,.86);
           border: 1px solid hsl(var(--h), var(--s), 30%); padding: 10px; overflow-y: auto;
           font-size: 12px; line-height: 1.45; display: none; user-select: text; }
    #log.open { display: block; }
    .m { margin: 0 0 8px; white-space: pre-wrap; word-wrap: break-word; }
    .m.u { color: #9fd4ff; }
    .m.j { color: hsl(var(--h), var(--s), 68%); }
    #perm { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
            background: rgba(0,0,0,.65); z-index: 10; }
    #perm.open { display: flex; }
    #permbox { width: min(540px, 90vw); background: #0a0705; border: 1px solid hsl(var(--h), var(--s), 55%);
               box-shadow: 0 0 30px hsla(var(--h), var(--s), 50%, .5); padding: 22px; }
    #permtitle { margin: 0 0 12px; letter-spacing: 3px; font-size: 14px; }
    #permdetail { color: #fff; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
                  margin-bottom: 18px; user-select: text; }
    #permbtns { display: flex; gap: 12px; justify-content: flex-end; }
    .btn.allow { border-color: #58e07a; color: #58e07a; }
    .btn.deny { border-color: #ff6666; color: #ff6666; }
    .btn.access { min-width: 90px; }
    .btn.access[data-level="FULL"] { border-color: #58e07a; color: #58e07a; }
    .btn.access[data-level="FILES"] { border-color: #ffd76a; color: #ffd76a; }
    .btn.access[data-level="LOCKED"] { border-color: #ff8a8a; color: #ff8a8a; }
</style>
</head>
<body>    <canvas id="orb"></canvas>

<div class="hud" id="title">J.A.R.V.I.S</div>
<div class="hud" id="badges">
    <div>ACCESS: <b id="b-access">LOCKED</b></div>
    <div>MEMORY: <b id="b-mem">0</b></div>
</div>
<div class="hud" id="controls">
    <button class="btn access" id="btn-access" data-level="LOCKED">ACCESS</button>
    <button class="btn" id="btn-log">LOG</button>
</div>
<div class="hud" id="status">Standby</div>
<div class="hud" id="caption">Say "Hey Jarvis" to begin.</div>
<div class="hud" id="inputbar">
    <input id="cmd" placeholder="or type a command and press Enter" autocomplete="off">
    <button class="btn" id="btn-send">SEND</button>
</div>
<div class="hud" id="log"></div>

<div id="perm">
    <div id="permbox">
        <div id="permtitle">PERMISSION REQUIRED</div>
        <div id="permdetail"></div>
        <div id="permbtns">
            <button class="btn deny" id="deny">DENY</button>
            <button class="btn allow" id="allow">ALLOW</button>
        </div>
    </div>
</div>

<script>
(function () {
    "use strict";
    const $ = id => document.getElementById(id);
    const cv = $('orb'), ctx = cv.getContext('2d');
    let W = 0, H = 0, DPR = 1;

    function resize() {
        DPR = Math.min(window.devicePixelRatio || 1, 2);
        W = window.innerWidth; H = window.innerHeight;
        cv.width = Math.floor(W * DPR); cv.height = Math.floor(H * DPR);
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    window.addEventListener('resize', resize);
    resize();

    function api(fn, ...args) {
        try {
            if (window.pywebview && window.pywebview.api && window.pywebview.api[fn]) {
                return window.pywebview.api[fn](...args);
            }
        } catch (e) { console.error(e); }
    }

    /* ------------------------------------------------------------ theme */
    const theme = { name: 'gold', h: 36, s: 100 };
    const NB = 14;
    let dimStyle = [], hotStyle = [];
    function buildStyles() {
        dimStyle = []; hotStyle = [];
        for (let b = 0; b < NB; b++) {
            const a = (b + 1) / NB;
            dimStyle.push('hsla(' + theme.h + ',' + theme.s + '%,55%,' + (a * 0.85).toFixed(3) + ')');
            hotStyle.push('hsla(' + theme.h + ',' + Math.max(0, theme.s - 45) + '%,80%,' + (a * 0.95).toFixed(3) + ')');
        }
    }
    function hsl(l, a) { return 'hsla(' + theme.h + ',' + theme.s + '%,' + l + '%,' + a + ')'; }
    buildStyles();

    /* -------------------------------------------------------- particles */
    const rnd = (a, b) => a + Math.random() * (b - a);
    const gauss = () => (Math.random() + Math.random() + Math.random() + Math.random() - 2) / 2;
    const GA = Math.PI * (3 - Math.sqrt(5));
    const P = [];

    function addStatic(x, y, z, s, hot) { P.push({ k: 0, x: x, y: y, z: z, s: s, hot: hot, ph: Math.random() * 6.283 }); }

    // spiral-wound shell
    for (let i = 0; i < 1500; i++) {
        const yy = 1 - 2 * (i + 0.5) / 1500, rr = Math.sqrt(1 - yy * yy), th = GA * i * 2.6;
        const rad = 0.74 + 0.26 * Math.pow(Math.random(), 0.6);
        addStatic(Math.cos(th) * rr * rad, yy * rad, Math.sin(th) * rr * rad, rnd(0.6, 1.5), Math.random() < 0.12 ? 1 : 0);
    }
    // bright core
    for (let i = 0; i < 520; i++) addStatic(gauss() * 0.26, gauss() * 0.26, gauss() * 0.26, rnd(0.7, 2.0), Math.random() < 0.55 ? 1 : 0);
    // outer dust
    for (let i = 0; i < 260; i++) {
        const u = rnd(0, 6.283), v = rnd(-1, 1), rr = Math.sqrt(1 - v * v), r = rnd(1.3, 2.0);
        addStatic(Math.cos(u) * rr * r, v * r, Math.sin(u) * rr * r, rnd(0.5, 1.1), 0);
    }
    // orbit rings: radius, tilt X, tilt Z, angular speed
    [[0.98, 0.45, 0.10, 0.35], [1.10, -0.70, 0.30, -0.28], [1.24, 1.20, -0.20, 0.20], [0.88, -1.10, 0.50, -0.40]].forEach(function (c) {
        const tx = c[1], tz = c[2];
        for (let i = 0; i < 260; i++) {
            P.push({ k: 1, a: rnd(0, 6.283), r: c[0] + gauss() * 0.012, y0: gauss() * 0.012, w: c[3],
                     ct: Math.cos(tx), st: Math.sin(tx), cz: Math.cos(tz), sz: Math.sin(tz),
                     s: rnd(0.6, 1.4), hot: Math.random() < 0.2 ? 1 : 0, ph: Math.random() * 6.283 });
        }
    });
    // spiral arms sweeping out from the core
    for (let arm = 0; arm < 5; arm++) {
        for (let i = 0; i < 150; i++) {
            const u = Math.pow(i / 150, 0.85);
            P.push({ k: 2, a: arm * 6.283 / 5 + u * 4.4 + gauss() * 0.06, r: 0.18 + u * 0.95,
                     y0: gauss() * 0.10 * (1 - u * 0.6), w: 0.12,
                     ct: Math.cos(0.5), st: Math.sin(0.5), cz: 1, sz: 0,
                     s: rnd(0.6, 1.6) * (1.1 - u * 0.5), hot: Math.random() < 0.3 ? 1 : 0, ph: Math.random() * 6.283 });
        }
    }

    const N = P.length;
    const SX = new Float32Array(N), SY = new Float32Array(N), SS = new Float32Array(N);
    const buckets = [];
    for (let b = 0; b < NB * 2; b++) buckets.push([]);

    /* ------------------------------------------------------------ state */
    let mode = 'idle', level = 0, targetLevel = 0.05, micLevel = 0;
    let rotY = 0, rotX = -0.35, vy = 0, vx = 0, zoom = 1, drag = false, lx = 0, ly = 0;
    let spinMul = 1, step = 1, emaDt = 0.016, last = 0, lastAdj = 0;

    function update(t, dt) {
        let spin = 0.0035; spinMul = 1;
        if (mode === 'thinking') { spin = 0.015; spinMul = 3.5; targetLevel = 0.22 + 0.12 * Math.sin(t * 5); }
        else if (mode === 'speaking') {
            spin = 0.007; spinMul = 1.7;
            targetLevel = Math.max(0, 0.34 + 0.28 * Math.sin(t * 7.3) * Math.sin(t * 2.9 + 1) + 0.10 * Math.sin(t * 17));
        }
        else if (mode === 'listening') { spin = 0.005; spinMul = 1.3; micLevel *= 0.92; targetLevel = 0.10 + micLevel * 0.9; }
        else { targetLevel = 0.04 + 0.025 * Math.sin(t * 1.2); }
        level += (targetLevel - level) * Math.min(1, dt * 9);
        if (!drag) {
            rotY += spin + vy; vy *= 0.94;
            rotX += vx; vx *= 0.90;
            rotX += (-0.35 - rotX) * 0.004;
        }
        // adaptive quality: if the machine can't keep up, draw fewer particles
        emaDt += (dt - emaDt) * 0.05;
        if (t - lastAdj > 2 && emaDt > 0.045 && step < 3) { step++; lastAdj = t; }
    }

    function render(t) {
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'rgba(4,3,2,0.34)';            // short trails = soft glow
        ctx.fillRect(0, 0, W, H);

        const cx = W / 2, cy = H / 2 - 30;
        const R = Math.min(W, H) * 0.28 * zoom * (1 + 0.025 * Math.sin(t * 1.3) + level * 0.10);

        ctx.globalCompositeOperation = 'lighter';
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * (1.25 + level * 0.5));
        g.addColorStop(0, hsl(70, 0.55 + level * 0.35));
        g.addColorStop(0.25, hsl(55, 0.22));
        g.addColorStop(1, hsl(40, 0));
        ctx.fillStyle = g;
        ctx.fillRect(cx - R * 2, cy - R * 2, R * 4, R * 4);

        const cY = Math.cos(rotY), sY = Math.sin(rotY), cX = Math.cos(rotX), sX = Math.sin(rotX);
        for (let b = 0; b < buckets.length; b++) buckets[b].length = 0;

        for (let i = 0; i < N; i += step) {
            const p = P[i];
            let x, y, z;
            if (p.k === 0) { x = p.x; y = p.y; z = p.z; }
            else {
                const a = p.a + p.w * t * spinMul;
                const x0 = Math.cos(a) * p.r, z0 = Math.sin(a) * p.r;
                const yy = p.y0 * p.ct - z0 * p.st;
                z = p.y0 * p.st + z0 * p.ct;
                x = x0 * p.cz - yy * p.sz;
                y = x0 * p.sz + yy * p.cz;
            }
            const wob = 1 + level * 0.22 * Math.sin(p.ph + t * 7 + (p.k === 0 ? p.y : p.y0) * 5);
            x *= wob; y *= wob; z *= wob;

            const x1 = x * cY + z * sY, z1 = -x * sY + z * cY;
            const y2 = y * cX - z1 * sX, z2 = y * sX + z1 * cX;
            const f = 1 / (1 - z2 * 0.22);
            SX[i] = cx + x1 * R * f;
            SY[i] = cy + y2 * R * f;
            SS[i] = Math.max(0.8, p.s * f * (R / 200) * (1 + level * 0.5)) * (step > 1 ? 1.4 : 1);

            const d = (z2 + 2) / 4;
            const bi = Math.floor(Math.max(0, Math.min(0.999, d * d + 0.12)) * NB);
            buckets[p.hot ? NB + bi : bi].push(i);
        }
        for (let b = 0; b < buckets.length; b++) {
            const arr = buckets[b];
            if (!arr.length) continue;
            ctx.fillStyle = b < NB ? dimStyle[b] : hotStyle[b - NB];
            for (let j = 0; j < arr.length; j++) {
                const i = arr[j], s = SS[i];
                ctx.fillRect(SX[i] - s / 2, SY[i] - s / 2, s, s);
            }
        }
        drawHud(t, cx, cy, R);
    }

    function drawHud(t, cx, cy, R) {
        ctx.globalCompositeOperation = 'lighter';
        const fast = mode === 'thinking' ? 4 : 1;

        // rotating tick ring
        const rt = R * 1.62;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(t * 0.05 * fast);
        ctx.lineWidth = 1;
        ctx.strokeStyle = hsl(55, 0.35);
        ctx.beginPath();
        for (let i = 0; i < 120; i++) {
            const a = i / 120 * 6.2832, l = (i % 10 === 0) ? 12 : (i % 5 === 0) ? 8 : 4;
            ctx.moveTo(Math.cos(a) * rt, Math.sin(a) * rt);
            ctx.lineTo(Math.cos(a) * (rt + l), Math.sin(a) * (rt + l));
        }
        ctx.stroke();
        ctx.restore();

        // sweeping arcs
        const arcs = [[1.45, 0.8, 2.2, 0.12, 0.55], [1.52, 3.4, 1.4, -0.18, 0.5], [1.38, 5.2, 1.1, 0.25, 0.45], [1.75, 1.9, 0.7, -0.07, 0.4]];
        ctx.lineWidth = 2;
        for (let i = 0; i < arcs.length; i++) {
            const c = arcs[i], r = R * c[0], a0 = c[1] + t * c[3] * fast;
            ctx.strokeStyle = hsl(60, c[4] * (0.7 + 0.3 * Math.sin(t * 2 + c[1])));
            ctx.beginPath();
            ctx.arc(cx, cy, r, a0, a0 + c[2]);
            ctx.stroke();
        }

        // frame corners + edge ruler marks
        const m = 14, L = 34;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = hsl(55, 0.6);
        const corners = [[m, m, 1, 1], [W - m, m, -1, 1], [m, H - m, 1, -1], [W - m, H - m, -1, -1]];
        for (let i = 0; i < corners.length; i++) {
            const c = corners[i];
            ctx.beginPath();
            ctx.moveTo(c[0] + c[2] * L, c[1]);
            ctx.lineTo(c[0], c[1]);
            ctx.lineTo(c[0], c[1] + c[3] * L);
            ctx.stroke();
        }
        ctx.fillStyle = hsl(55, 0.5);
        ctx.fillRect(m, cy, 26, 1); ctx.fillRect(W - m - 26, cy, 26, 1);
        ctx.fillRect(cx - 1, m + 44, 1, 14); ctx.fillRect(cx - 1, H - m - 150, 1, 10);

        // small corner widget (bottom-right)
        const wx = W - 46, wy = H - 46;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = hsl(60, 0.7);
        ctx.beginPath(); ctx.arc(wx, wy, 11, 0, 6.2832); ctx.stroke();
        ctx.beginPath(); ctx.arc(wx, wy, 5, t * 2, t * 2 + 4); ctx.stroke();
    }

    let rafId = 0;
    function frame(ts) {
        try {
            const t = ts / 1000;
            const dt = Math.min(0.05, (t - last) || 0.016);
            last = t;
            update(t, dt);
            render(t);
        } catch (e) { console.error(e); }
        rafId = requestAnimationFrame(frame);
    }
    rafId = requestAnimationFrame(frame);

    /* ------------------------------------------------- 3D interaction */
    cv.addEventListener('pointerdown', function (e) {
        drag = true; lx = e.clientX; ly = e.clientY; cv.classList.add('drag');
        try { cv.setPointerCapture(e.pointerId); } catch (err) {}
    });
    cv.addEventListener('pointermove', function (e) {
        if (!drag) return;
        const dx = e.clientX - lx, dy = e.clientY - ly;
        lx = e.clientX; ly = e.clientY;
        rotY += dx * 0.008; rotX += dy * 0.008;
        vy = dx * 0.0012; vx = dy * 0.0006;
        rotX = Math.max(-1.3, Math.min(1.3, rotX));
    });
    const endDrag = function () { drag = false; cv.classList.remove('drag'); };
    cv.addEventListener('pointerup', endDrag);
    cv.addEventListener('pointercancel', endDrag);
    cv.addEventListener('wheel', function (e) {
        e.preventDefault();
        zoom = Math.max(0.55, Math.min(1.9, zoom * (e.deltaY < 0 ? 1.06 : 0.94)));
    }, { passive: false });
    cv.addEventListener('dblclick', function () { zoom = 1; rotX = -0.35; });

    /* ------------------------------------------------ text / log / status */
    const logEl = $('log'), cap = $('caption');
    function tail(text) { return text.length > 230 ? '…' + text.slice(-230) : text; }
    function pushLog(cls, text) {
        const d = document.createElement('div');
        d.className = 'm ' + cls;
        d.innerText = text;
        logEl.appendChild(d);
        logEl.scrollTop = logEl.scrollHeight;
        return d;
    }
    let liveEl = null;

    window.updateStatus = function (text) {
        $('status').innerText = text;
        const l = String(text).toLowerCase();
        if (/listen|awaiting/.test(l)) mode = 'listening';
        else if (/transcrib|think/.test(l)) mode = 'thinking';
        else if (/speak|playing|launch|brows/.test(l)) mode = 'speaking';
        else mode = 'idle';
    };
    window.addMessage = function (sender, text) {
        if (sender === 'user') { pushLog('u', 'You: ' + text); cap.innerText = '› ' + tail(text); }
        else { pushLog('j', 'JARVIS: ' + text); cap.innerText = tail(text); }
    };
    window.streamJarvis = function (text) {
        if (!liveEl) liveEl = pushLog('j', '');
        liveEl.innerText = 'JARVIS: ' + text;
        logEl.scrollTop = logEl.scrollHeight;
        cap.innerText = tail(text);
    };
    window.endStream = function () { liveEl = null; };
    window.setLevel = function (x) { micLevel = Math.max(0, Math.min(1, Number(x) || 0)); };
    window.setMemory = function (n) { $('b-mem').innerText = String(n); };
    window.setAccess = function (level) {
        const el = $('b-access');
        el.innerText = level;
        el.style.color = level === 'FULL' ? '#7dff9a' : level === 'FILES' ? '#ffd76a' : '#ff8a8a';
        const btn = $('btn-access');
        if (btn) { btn.dataset.level = level; btn.innerText = level; }
    };

    window.setColorsVisible = function (on) {
        // colors are purely voice-driven now — the swatch dots and the button
        // are no longer present in the page, so this is a no-op kept only so the
        // Python side can keep calling it without error during upgrades.
    };

    /* ------------------------------------------------------------ colors */
    let themes = {};
    function markActive() {
        // swatches are gone; keep the empty function for parity with css helpers
    }
    window.initThemes = function (obj) {
        themes = obj;
    };
    window.setThemeHSL = function (name, h, s) {
        theme.name = name; theme.h = h; theme.s = s;
        document.documentElement.style.setProperty('--h', h);
        document.documentElement.style.setProperty('--s', s + '%');
        buildStyles();
    };

    /* -------------------------------------------------- permission dialog */
    let permId = null, permTimer = null;
    function answer(ok) {
        if (permId === null) return;
        const id = permId;
        window.hidePermission();
        api('permission_response', id, ok);
    }
    window.showPermission = function (id, title, detail) {
        permId = id;
        $('permtitle').innerText = title;
        $('permdetail').innerText = detail;
        $('perm').classList.add('open');
        clearTimeout(permTimer);
        permTimer = setTimeout(function () { answer(false); }, 46000);
    };
    window.hidePermission = function () {
        $('perm').classList.remove('open');
        permId = null;
        clearTimeout(permTimer);
    };
    $('allow').onclick = function () { answer(true); };
    $('deny').onclick = function () { answer(false); };

    /* --------------------------------------------------------- controls */
    const cmd = $('cmd');
    function send() {
        const v = cmd.value.trim();
        if (!v) return;
        cmd.value = '';
        api('submit_text', v);
    }
    $('btn-send').onclick = send;
    cmd.addEventListener('keydown', function (e) { if (e.key === 'Enter') send(); });
    $('btn-log').onclick = function () { logEl.classList.toggle('open'); };
    $('btn-access').onclick = function () {
        try { if (window.pywebview && window.pywebview.api) window.pywebview.api.cycle_access(); } catch (e) {}
    };
})();
</script>
</body>
</html>
"""

def _default_system_prompt():
    return (
        "You are JARVIS, a highly intelligent, witty, and friendly AI companion. "
        "Answer naturally and concisely in 1 to 3 sentences. "
        "Use plain sentences only: no markdown, lists, emojis or code blocks, because your reply is read aloud. "
        "You have a persistent memory."
    )


chat_memory = [
    {"role": "system", "content": _default_system_prompt()}
]


# --------------------------------------------------------------- helpers ----
def ui_status(window, text):
    try:
        window.evaluate_js(f"updateStatus({json.dumps(text)});")
    except Exception as e:
        print(f"UI status error: {e}")


def ui_message(window, sender, text):
    try:
        window.evaluate_js(f"addMessage({json.dumps(sender)}, {json.dumps(text)});")
    except Exception as e:
        print(f"UI message error: {e}")


def ui_stream(window, text):
    try:
        window.evaluate_js(f"streamJarvis({json.dumps(text)});")
    except Exception as e:
        print(f"UI stream error: {e}")


def ui_stream_end(window):
    try:
        window.evaluate_js("endStream();")
    except Exception:
        pass


def normalize(text):
    """Lowercase and strip punctuation. Whisper returns 'Stop.' / 'Thank you.'
    so exact-match checks against plain strings never worked."""
    return re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE).strip()


# Words that are part of the *command*, not the thing to search for
FILLER_WORDS = {
    "open", "launch", "play", "search", "youtube", "on", "in", "and", "then",
    "please", "can", "could", "you", "will", "would", "me", "my", "i", "want",
    "to", "for", "some", "a", "an", "the", "hey", "jarvis", "of",
    "start", "run", "close", "quit", "exit", "kill", "app", "application", "program",
    "okay", "ok", "just", "thats", "that", "its", "it", "so", "well", "yeah", "actually", "at", "is",
}


def clean_query(norm):
    """'open youtube and play video songs tamil video songs' -> 'video songs tamil'.
    Removes command/filler words, keeps meaningful ones (song, video, language...),
    and drops repeated words caused by the speaker repeating themselves."""
    words = []
    for w in norm.split():
        if w in FILLER_WORDS or w in words:
            continue
        words.append(w)
    return " ".join(words)


LEAD_WORDS = {"hey", "jarvis", "please", "can", "could", "will", "would", "you", "we", "lets",
              "let", "us", "just", "now", "okay", "ok", "so", "actually"}
OPEN_VERBS = {"open", "launch", "start", "run"}
CLOSE_VERBS = {"close", "closed", "quit", "exit", "kill", "terminate", "shutdown", "force"}
END_VERBS = {"close", "closed", "end", "stop", "quit", "exit", "finish", "terminate", "shutdown", "kill"}
CHAT_WORDS = {"chat", "conversation", "session"}


def parse_app_command(norm):
    """Only treat a sentence as 'open/close <app>' when it STARTS with the verb
    (after polite lead-ins like 'can you'). A verb buried mid-sentence
    ('...you to close the chat because...') is normal conversation."""
    words = norm.split()
    i = 0
    while i < len(words) and words[i] in LEAD_WORDS:
        i += 1
    if i >= len(words):
        return None
    verb = words[i]
    if verb in OPEN_VERBS:
        kind = "open"
    elif verb in CLOSE_VERBS:
        kind = "close"
    else:
        return None
    name = [w for w in words[i + 1:] if w not in FILLER_WORDS]
    return kind, " ".join(name)


def _is_jarvis_exit(norm):
    """Returns True if the user wants to shut down JARVIS entirely (not just end the session)."""
    # Direct phrase match
    if norm in JARVIS_EXIT_PHRASES:
        return True
    # Fuzzy: any exit verb + "jarvis" within a short sentence
    words = norm.split()
    has_exit_verb = bool(END_VERBS & set(words))
    has_jarvis = any(
        difflib.SequenceMatcher(None, w, "jarvis").ratio() >= 0.75
        for w in words
    )
    return has_exit_verb and has_jarvis and len(words) <= 5


def is_end_session(norm):
    """'close this chat', 'end the conversation', 'close Jarvis' -> end the voice session."""
    words = norm.split()
    if not words:
        return False
    if END_VERBS & set(words) and CHAT_WORDS & set(words):
        return True
    i = 0
    while i < len(words) and words[i] in LEAD_WORDS - {"jarvis"}:
        i += 1
    if i < len(words) and words[i] in END_VERBS:
        rest = "".join(words[i + 1:])
        # Whisper hears "Jarvis" as "jar vis", "jar vca", "service"...
        if rest and len(words) - i - 1 <= 2 and difflib.SequenceMatcher(None, rest, "jarvis").ratio() >= 0.6:
            return True
    return False


def flush_mic(mic_stream):
    """Discard audio that piled up while we were transcribing/speaking, so we
    don't record stale audio (or JARVIS's own voice)."""
    try:
        while mic_stream.get_read_available() >= CHUNK:
            mic_stream.read(CHUNK, exception_on_overflow=False)
    except Exception:
        pass


def _play_beep():
    """Play a short earcon beep so the user knows JARVIS heard the wake word.
    Generates a sine-wave WAV in memory and plays it with aplay (non-blocking)."""
    try:
        import struct, math
        sr = 22050
        n_samples = int(sr * WAKE_BEEP_MS / 1000)
        fade = min(64, n_samples // 4)
        samples = []
        for i in range(n_samples):
            amp = 0.6
            # linear fade-in and fade-out to avoid clicks
            if i < fade:
                amp *= i / fade
            elif i >= n_samples - fade:
                amp *= (n_samples - i) / fade
            samples.append(int(amp * 32767 * math.sin(2 * math.pi * WAKE_BEEP_HZ * i / sr)))
        beep_path = "/tmp/jarvis_beep.wav"
        with wave.open(beep_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
        subprocess.Popen(["aplay", "-q", beep_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Beep error: {e}")


def record_audio(mic_stream, max_seconds=20, silence_seconds=0.7, start_timeout=8.0):
    """Record until the speaker stops talking, so long sentences aren't cut off."""
    flush_mic(mic_stream)
    chunk_sec = CHUNK / RATE
    frames, floor = [], []
    speech_started, silent_chunks = False, 0

    for i in range(int(max_seconds / chunk_sec)):
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        ui_level_push(rms)
        # Collect 12 chunks for a stable noise floor; skip first 2 (mic warm-up transients)
        if 2 <= i < 14:
            floor.append(rms)
        threshold = min(max(MIN_SPEECH_RMS, 2.5 * min(floor or [MIN_SPEECH_RMS])), MAX_SPEECH_RMS)

        if rms > threshold:
            speech_started, silent_chunks = True, 0
        elif speech_started:
            silent_chunks += 1
            if silent_chunks * chunk_sec >= silence_seconds:
                break  # they stopped talking
        elif (i + 1) * chunk_sec >= start_timeout:
            break  # nobody said anything

    ui_level_push(0)
    if not speech_started:
        # Nobody actually spoke - only background noise. Returning a file here
        # makes Whisper hallucinate a sentence (usually echoing our own prompt),
        # which then gets executed as a command. Signal "no speech" instead.
        return None
    with wave.open(CMD_WAV, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    return CMD_WAV


# Whisper invents these when it is fed silence or noise instead of speech.
_HALLUCINATIONS = {
    # Classic Whisper noise / silence outputs
    "you", "thank you.", "thanks for watching!", "thanks for watching.",
    "please subscribe", "subscribe to my channel", "subtitles by", "amara.org",
    "transcription by", "...", ".", "- -", "okay.", "bye.", "hmm",
    # Common additional Whisper hallucinations
    "mm-hmm.", "mm-hmm", "mmm.", "mmm", "mhm.", "mhm",
    "i'll see you next time.", "i'll see you next time",
    "see you later.", "see you later!", "see you next time.",
    "thank you for watching.", "thanks for tuning in.",
    "don't forget to like and subscribe.",
    "this video is brought to you by",
    "[music]", "[ music ]", "[applause]", "[laughter]",
    "[silence]", "[ silence ]", "(silence)",
    "subtitles by the amara.org community",
    "i'll be right back.", "stay tuned.",
    "brought to you by", "sponsored by",
    "thank you for listening.",
    "[inaudible]", "( inaudible )",
    "caption by", "captions by",
}
_PROMPT_ECHO = "desktop assistant named jarvis"


def _strip_hallucination(text):
    """Drop transcript text that is really Whisper filling in silence.
    Returns "" when the text looks like an artifact, else the cleaned text."""
    t = text.strip()
    if not t:
        return ""
    low = t.lower().strip(" .!?,-\u2026")
    if not low or low in _HALLUCINATIONS:
        return ""
    # our own initial_prompt coming back verbatim
    if _PROMPT_ECHO in low or low.startswith("voice commands for a desktop assistant"):
        return ""
    if low.strip(".!") == WHISPER_PROMPT.lower().strip(".!"):
        return ""
    for junk in ("thanks for watching", "subscribe", "amara.org", "subtitles by", "transcription by", "www."):
        if junk in low:
            return ""
    return t


_current_voice = {"name": "lessac"}
_current_personality = {"name": "jarvis"}


def _voice_path(name):
    return os.path.join(JARVIS_DIR_ABS, VOICE_MODELS[name]["file"])


def _voice_cfg_path(name):
    return _voice_path(name) + ".json"


def voice_available(name):
    """True if the model file exists locally (and its config)."""
    return os.path.isfile(_voice_path(name)) and os.path.isfile(_voice_cfg_path(name))


def voice_list():
    """[(name, gender, label, available)] for each registered voice."""
    return [(n, VOICE_MODELS[n]["gender"], VOICE_MODELS[n]["label"], voice_available(n))
            for n in VOICE_MODELS]


def current_voice():
    return _current_voice["name"]


def set_voice(name, download_if_missing=False, window=None, persist=True):
    """Switch to a registered voice. Returns (ok, message)."""
    alias = VOICE_ALIASES.get(name, name)
    if alias not in VOICE_MODELS:
        return False, f"I don't have a voice called {name}. Try: " + ", ".join(VOICE_MODELS) + "; or say male/female."
    if not voice_available(alias):
        if not download_if_missing:
            return (False,
                    f"The {alias} voice needs to be downloaded first. Say: download voice {alias}, or switch to lessac which is ready.")
        if window is not None:
            ui_status(window, "Downloading voice...")
            respond(window, f"Downloading the {alias} voice. This takes a few seconds, sir.", "Downloading voice...", log=False)
        ok = _download_voice(alias)
        if not ok:
            return False, f"Download failed for voice {alias}. Try again later or use lessac."
    _current_voice["name"] = alias
    if persist:
        cfg = _read_json(CONFIG_PATH, {})
        cfg["voice"] = alias
        _write_json(CONFIG_PATH, cfg)
    label = VOICE_MODELS[alias]["label"]
    return True, f"Voice changed to {label}."


def _download_voice(name):
    """Download the onnx + json for a voice. Returns True on success."""
    info = VOICE_MODELS.get(name)
    if not info or not info.get("hf"):
        return False
    ok = True
    for url_key, path in (("hf", _voice_path(name)), ("cfg_hf", _voice_cfg_path(name))):
        url = info.get(url_key)
        if not url:
            continue
        tmp = path + ".part"
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(path):
                os.remove(path)
            os.replace(tmp, path)
        except Exception as e:
            print(f"Voice download failed ({url}): {e}")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            ok = False
    return ok and voice_available(name)


def _espeak_fallback(text, gender):
    """espeak-ng / espeak fallback when piper fails (or model missing)."""
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            voice = "en-us+m3" if gender == "male" else "en-us+f3"
            try:
                subprocess.run([exe, "-v", voice, "-s", 170, text],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=15)
                return True
            except Exception:
                continue
    return False


def _piper_bin():
    """piper on PATH, or the one shipped in this venv (works even when the
    venv wasn't activated, e.g. running `python jarvis.py` directly)."""
    exe = shutil.which("piper")
    if exe:
        return exe
    cand = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "piper")
    return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None


def speak(text):
    """Blocking TTS. Uses currently selected piper voice; falls back to espeak-ng gendered voice.
    Always waits TTS_COOLDOWN seconds after playback ends before returning, so the mic
    cannot pick up JARVIS's own voice and cause a phantom wake-word / command trigger."""
    text = re.sub(r"[*_`#]", "", text)
    text = " ".join(text.split())
    if not text:
        return
    name = _current_voice["name"]
    gender = VOICE_MODELS[name]["gender"]
    try:
        piper = _piper_bin()
        if piper and voice_available(name):
            subprocess.run(
                [piper, "--model", _voice_path(name), "--config", _voice_cfg_path(name),
                 "--output_file", TTS_WAV],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if os.path.exists(TTS_WAV) and os.path.getsize(TTS_WAV) > 512:
                subprocess.run(["aplay", "-q", TTS_WAV], stderr=subprocess.DEVNULL, check=False)
                try: os.remove(TTS_WAV)
                except: pass
                time.sleep(TTS_COOLDOWN)   # mic cooldown: prevent self-trigger
                return
        _espeak_fallback(text, gender)
    except Exception as e:
        print(f"TTS Error (voice={name}): {e}")
        _espeak_fallback(text, gender)
    time.sleep(TTS_COOLDOWN)   # cooldown even on fallback path


# ---------- personality / buddy mode ----------
def current_personality():
    return _current_personality["name"]


def set_personality(name, persist=True):
    if name not in PERSONALITY_MODES:
        return False, f"Available modes: " + ", ".join(PERSONALITY_MODES) + "."
    _current_personality["name"] = name
    if persist:
        cfg = _read_json(CONFIG_PATH, {})
        cfg["personality"] = name
        _write_json(CONFIG_PATH, cfg)
    return True, f"Mode: {PERSONALITY_MODES[name]['label']}."


def personality_tokens(base, fast_mode):
    mode = PERSONALITY_MODES[current_personality()]
    ceiling = mode["tokens"]
    n = base if not fast_mode else max(70, base - 60)
    return min(n, ceiling)


def respond(window, reply, status, log=True):
    print(f"JARVIS: {reply}")
    ui_message(window, "jarvis", reply)
    if log:
        mem_log_action(reply)          # remembered, so JARVIS can say what it did earlier
    ui_status(window, "Speaking...")   # orb animates while the voice plays
    speak(reply)                       # blocks until finished
    ui_status(window, status)


player_proc = None


def stop_player():
    global player_proc
    if player_proc and player_proc.poll() is None:
        player_proc.terminate()
    player_proc = None


def play_on_youtube(query):
    """Play the first YouTube result directly in mpv, so it starts immediately.
    (Browsers block autoplay for pages opened by a script.)"""
    global player_proc
    if shutil.which("mpv") and shutil.which("yt-dlp"):
        stop_player()
        player_proc = subprocess.Popen(
            ["mpv", "--force-window=yes", f"--input-ipc-server={MPV_SOCK}",
             "--ytdl-format=bv*[height<=720]+ba/b[height<=720]/best",
             f"ytdl://ytsearch10:{query}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return
    print("mpv / yt-dlp not installed; falling back to the browser (autoplay may be blocked)")
    try:
        import pywhatkit  # lazy import: it fails at import time if offline
        pywhatkit.playonyt(query)
    except Exception as e:
        print(f"pywhatkit failed ({e}); falling back to a YouTube search")
        open_url_browser(
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")


# ------------------------------------------------------ installed apps ----
_app_index = None

DESKTOP_DIRS = [
    "/usr/share/applications",
    "/usr/local/share/applications",
    "/var/lib/flatpak/exports/share/applications",
    "/var/lib/snapd/desktop/applications",
    os.path.expanduser("~/.local/share/applications"),
    os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
]


def build_app_index():
    """Read every installed application's .desktop file -> {spoken-style name: entry}."""
    entries = []
    for d in DESKTOP_DIRS:
        for path in glob.glob(os.path.join(d, "*.desktop")):
            cp = configparser.RawConfigParser(strict=False, interpolation=None)
            cp.optionxform = str  # keep key case ("Name", "Exec")
            try:
                cp.read(path, encoding="utf-8")
                sec = cp["Desktop Entry"]
            except Exception:
                continue
            if sec.get("Type") != "Application" or not sec.get("Exec"):
                continue
            if sec.get("NoDisplay", "").lower() == "true" or sec.get("Hidden", "").lower() == "true":
                continue
            stem = os.path.basename(path)[:-len(".desktop")]
            entries.append({
                "name": sec.get("Name", stem),
                "generic": sec.get("GenericName", ""),
                "id": stem,
                "exec": sec["Exec"],
            })

    index = {}
    for e in entries:                       # real names win...
        index.setdefault(normalize(e["name"]), e)
    for e in entries:                       # ...then generic names and ids
        if e["generic"]:
            index.setdefault(normalize(e["generic"]), e)
        index.setdefault(normalize(re.sub(r"[._-]", " ", e["id"])), e)
    print(f"Indexed {len(entries)} installed applications")
    return index


def get_app_index():
    global _app_index
    if _app_index is None:
        _app_index = build_app_index()
    return _app_index


def find_app(spoken):
    index = get_app_index()
    keys = list(index)
    for cand in dict.fromkeys([ALIASES.get(spoken, spoken), spoken]):
        if cand in index:
            return index[cand]
        words = cand.split()
        hits = [k for k in keys if all(w in k.split() for w in words)]
        if hits:
            return index[min(hits, key=len)]
        close = difflib.get_close_matches(cand, keys, n=1, cutoff=0.7)  # tolerates Whisper mishearings
        if close:
            return index[close[0]]
        first = {k.split()[0]: k for k in keys}  # "telegran" -> "telegram desktop"
        close = difflib.get_close_matches(cand, list(first), n=1, cutoff=0.75)
        if close:
            return index[first[close[0]]]
    return None


def launch_app(entry):
    if shutil.which("gtk-launch"):
        cmd = ["gtk-launch", entry["id"]]
    else:
        cmd = shlex.split(re.sub(r"%[a-zA-Z%]", "", entry["exec"]))
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def _kill_by_name(name):
    """Try progressively broader kill strategies.
    Returns True if at least one process was terminated.
    """
    name = name.strip()
    if not name:
        return False

    # Strategy 1: exact binary name match (fastest, safest)
    r = subprocess.run(["pkill", "-x", name[:15]], capture_output=True)
    if r.returncode == 0:
        return True

    # Strategy 2: full command-line substring match (catches wrappers / args)
    r = subprocess.run(["pkill", "-f", re.escape(name)], capture_output=True)
    if r.returncode == 0:
        return True

    # Strategy 3: scan /proc for any process whose comm or cmdline contains the name
    killed = False
    try:
        for pid_dir in os.listdir("/proc"):
            if not pid_dir.isdigit():
                continue
            try:
                comm_path = f"/proc/{pid_dir}/comm"
                cmd_path  = f"/proc/{pid_dir}/cmdline"
                comm = open(comm_path).read().strip() if os.path.exists(comm_path) else ""
                cmd  = open(cmd_path).read().replace("\x00", " ").strip() if os.path.exists(cmd_path) else ""
                if (name.lower() in comm.lower() or name.lower() in cmd.lower()):
                    os.kill(int(pid_dir), 15)   # SIGTERM first
                    time.sleep(0.15)
                    try:
                        os.kill(int(pid_dir), 9)   # SIGKILL if still alive
                    except ProcessLookupError:
                        pass                        # already gone — good
                    killed = True
            except (PermissionError, FileNotFoundError, ValueError):
                continue
    except Exception:
        pass
    return killed


def close_app(entry):
    """Terminate the app associated with a .desktop entry.
    Uses a cascade of kill strategies so background/Flatpak/Snap processes
    are reliably found and terminated even when pkill -x misses them.
    """
    # Derive candidate process names from the .desktop Exec field
    candidates = set()
    try:
        raw_exec = re.sub(r"%[a-zA-Z%]", "", entry.get("exec", ""))
        exe = os.path.basename(shlex.split(raw_exec)[0])
        if exe not in {"flatpak", "snap", "env", "sh", "bash", "python", "python3"}:
            candidates.add(exe)
        # Also try the app id stem (e.g. "org.mozilla.firefox" → "firefox")
        app_id = entry.get("id", "")
        stem = app_id.split(".")[-1].lower()
        if stem and stem not in {"app", "desktop"}:
            candidates.add(stem)
        # And the display name lowercased
        display = normalize(entry.get("name", "")).split()[0]
        if display:
            candidates.add(display)
    except Exception:
        pass

    if not candidates:
        return False

    killed = any(_kill_by_name(c) for c in candidates)
    return killed


# ---------------------------------------------------- typing / dictation ----
_typing_mode = {"on": False}

TYPING_ON_PHRASES = {"start typing", "typing mode", "typing mode on", "start dictation",
                     "dictation mode", "dictation mode on", "type mode", "start typing mode"}
TYPING_OFF_PHRASES = {"stop typing", "stop typing mode", "typing mode off", "end typing",
                      "stop dictation", "end dictation", "dictation mode off", "type mode off"}
TYPE_RE = re.compile(r"^\W*(?:hey jarvis\W+)?(?:please\s+)?type(?:\s+that)?\s*[:,]?\s+(.+?)\s*$",
                     re.I | re.S)


def typing_mode_on():
    return _typing_mode["on"]


def _type_text(text):
    """Type `text` into whatever window currently has focus with human-like variable speed.

    Strategy:
    - Types word-by-word so xdotool startup overhead is paid only once per word.
    - Inter-character delay varies randomly (35-100 ms) to feel organic.
    - Inter-word pause varies randomly (60-250 ms) to mimic natural cadence.
    Prefers xdotool; falls back to pyautogui.
    """
    import random
    text = " ".join(str(text).split())              # one line, collapsed whitespace
    text = "".join(ch for ch in text if 32 <= ord(ch) < 127)   # X11 key events are ASCII-only
    if not text:
        return False

    if shutil.which("xdotool"):
        words = text.split(" ")
        ok = True
        for idx, word in enumerate(words):
            chunk = word if idx == 0 else " " + word
            delay = random.randint(35, 100)   # ms between keystrokes — human range
            r = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", str(delay), chunk],
                capture_output=True
            )
            if r.returncode != 0:
                ok = False
            if idx < len(words) - 1:
                time.sleep(random.uniform(0.06, 0.25))   # inter-word pause
        return ok

    try:
        import pyautogui
        import random as _r
        for char in text:
            pyautogui.write(char, interval=_r.uniform(0.035, 0.10), _pause=False)
        return True
    except Exception as e:
        print(f"Typing failed: {e}")
        return False


def handle_typing_command(window, norm, command):
    """'type <text>' types once; 'start typing' begins dictation mode where every
   thing said is typed into the focused window until 'stop typing' / 'stop'."""
    # one-shot: "type hello world"
    if not _typing_mode["on"]:
        m = TYPE_RE.match(command)
        if m and norm.split()[0] != "mode":
            text = m.group(1).strip()
            if text and text.lower() not in {"mode", "this", "it"}:
                ui_status(window, "Typing...")
                ok = _type_text(text)
                ui_message(window, "jarvis", f"[typed] {text}" if ok else "[typing failed]")
                ui_status(window, "Listening for next command...")
                return True
        if norm in TYPING_ON_PHRASES:
            _typing_mode["on"] = True
            respond(window, "Typing mode on. Say anything and I will type it into the focused window. "
                            "Say stop typing when you are done.", "Standby", log=False)
            return True
        if norm in TYPING_OFF_PHRASES:      # idempotent: "stop typing" while already off
            respond(window, "Typing mode is already off.", "Standby", log=False)
            return True
        return False

    # dictation mode is active: everything gets typed, not executed
    if norm in TYPING_OFF_PHRASES or norm in STOP_PHRASES or is_end_session(norm):
        _typing_mode["on"] = False
        respond(window, "Typing mode off.", "Standby", log=False)
        return True
    m = TYPE_RE.match(command)
    text = m.group(1).strip() if m else command.strip()
    if text:
        ui_status(window, "Typing...")
        ok = _type_text(text)
        ui_message(window, "jarvis", f"[typed] {text}" if ok else "[typing failed]")
        ui_status(window, "Listening for next command...")
    return True


# ------------------------------------------------------------ web search ----
WEB_TARGETS = {"browser", "web", "internet", "online", "google", "chrome", "firefox",
               "www", "engine", "page", "site", "search"}
SEARCH_VERBS = {"search", "google", "browse"}


def handle_web_search(window, norm):
    """'search silent hill on browser', 'google python lists' -> open a real web search.
    Runs after the file search and the YouTube branch, so those keep priority."""
    words = norm.split()
    i = 0
    while i < len(words) and words[i] in LEAD_WORDS:
        i += 1
    rest = words[i:]
    if not rest:
        return False
    verb = rest[0]
    if verb == "look" and len(rest) > 1 and rest[1] == "up":
        rest = rest[2:]
    elif verb in SEARCH_VERBS:
        rest = rest[1:]
    else:
        return False
    qwords = [w for w in rest if w not in WEB_TARGETS and w not in FILLER_WORDS]
    seen = set()
    qwords = [w for w in qwords if not (w in seen or seen.add(w))]   # Whisper often repeats itself
    query = " ".join(qwords)
    if not query:
        respond(window, "What should I search for?", "Listening...")
        return True
    respond(window, f"Searching the web for {query}.", "Browsing...")
    return open_url_browser(
        f"https://www.google.com/search?q={urllib.parse.quote(query)}")
    # (returns True on success; on failure it has already spoken an error via
    # webbrowser/webbrowser fallback, so the call site doesn't need another branch)


# -------------------------------------------------------- command router ----
def handle_app_command(window, norm):
    """Returns True if the sentence was an app command and has been handled;
    False to let it fall through to the chat model."""
    cmd = parse_app_command(norm)
    if not cmd:
        return False
    kind, name = cmd
    if not name:
        respond(window, f"Which application should I {kind}?", "Listening...")
        return True

    entry = find_app(name)

    # ---- OPEN ----
    if kind == "open":
        if entry:
            respond(window, f"Opening {entry['name']}, sir.", "Launching App...")
            launch_app(entry)
            return True
        if open_site_by_name(name):
            respond(window, f"Opening {name}.", "Browsing...")
            return True
        if len(name.split()) <= 2:   # short -> clearly a command, tell the user
            respond(window, f"I couldn't find an application called {name}.", "App Not Found")
            return True
        return False   # long sentence -> probably conversation, let the LLM handle it

    # ---- CLOSE ----
    if kind == "close":
        if entry:
            if close_app(entry):
                respond(window, f"Closed {entry['name']}.", "Standby")
            else:
                # .desktop entry found but pkill missed — try killing by spoken name directly
                if _kill_by_name(name.split()[0]):
                    respond(window, f"Terminated {entry['name']}.", "Standby")
                else:
                    respond(window, f"I couldn't terminate {entry['name']}. "
                            "It may have already closed.", "Standby")
            return True
        # No .desktop entry — try killing by the spoken name anyway
        if len(name.split()) <= 2:
            if _kill_by_name(name.split()[0]):
                respond(window, f"Terminated {name}.", "Standby")
            else:
                respond(window, f"No process named {name} found.", "Standby")
            return True
        return False

    return False


# -------------------------------------------------- media & system control ----
NUMBER_WORDS = {"zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
                "half": 50, "full": 100, "max": 100}
MEDIA_VERBS = {"pause": "pause", "unpause": "play", "resume": "play", "continue": "play",
               "next": "next", "skip": "next", "previous": "previous", "play": "play"}
MEDIA_REST = {"the", "music", "song", "songs", "video", "videos", "track", "playback", "it",
              "this", "that", "one", "playing", "player", "again", "please", "now"}


def parse_number(words):
    for w in words:
        if w.isdigit():
            return min(int(w), 100)
        if w in NUMBER_WORDS:
            return NUMBER_WORDS[w]
    return None


def parse_media_command(norm):
    """'pause', 'next song', 'resume the music' -> 'pause' / 'next' / 'play' (else None)."""
    words = norm.split()
    i = 0
    while i < len(words) and words[i] in LEAD_WORDS:
        i += 1
    if i >= len(words) or words[i] not in MEDIA_VERBS:
        return None
    if not set(words[i + 1:]) <= MEDIA_REST:
        return None
    return MEDIA_VERBS[words[i]]


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except Exception:
        return None


def mpv_command(*cmd):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect(MPV_SOCK)
        sock.sendall(json.dumps({"command": list(cmd)}).encode() + b"\n")
    finally:
        sock.close()


def media_control(action):
    """True = sent, False = failed, None = nothing to control."""
    if player_proc and player_proc.poll() is None:  # our own mpv player
        cmd = {"pause": ["set_property", "pause", True], "play": ["set_property", "pause", False],
               "next": ["playlist-next"], "previous": ["playlist-prev"]}[action]
        try:
            mpv_command(*cmd)
            return True
        except Exception as e:
            print(f"mpv IPC error: {e}")
            return False
    if shutil.which("playerctl"):  # Spotify, VLC, Firefox... anything that supports MPRIS
        r = run(["playerctl", action])
        return True if r and r.returncode == 0 else None
    return None


def handle_media_command(window, norm):
    action = parse_media_command(norm)
    if not action:
        return False
    ok = media_control(action)
    if ok is None:
        if action == "play":
            return False  # bare "play ..." -> let the YouTube branch handle it
        respond(window, "Nothing is playing right now.", "Standby")
    elif ok is False:
        respond(window, "I couldn't control the player.", "Standby")
    else:
        respond(window, {"pause": "Paused.", "play": "Resuming.", "next": "Next track.",
                         "previous": "Previous track."}[action], "Standby")
    return True


def get_volume():
    r = run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    m = re.search(r"(\d+)%", r.stdout) if r else None
    return int(m.group(1)) if m else None


def set_volume(percent):
    percent = max(0, min(100, percent))
    r = run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
    return percent if r and r.returncode == 0 else None


def set_mute(mute):
    r = run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if mute else "0"])
    return bool(r and r.returncode == 0)


def get_brightness():
    if not shutil.which("brightnessctl"):
        return None
    r = run(["brightnessctl", "-m"])
    try:
        return int(r.stdout.strip().split(",")[3].rstrip("%"))
    except Exception:
        return None


def set_brightness(percent):
    percent = max(5, min(100, percent))  # never go fully black
    r = run(["brightnessctl", "set", f"{percent}%"])
    return percent if r and r.returncode == 0 else None


def battery_report():
    try:
        import psutil
        b = psutil.sensors_battery()
    except Exception:
        return "I need the psutil package to read the battery."
    if not b:
        return "I couldn't find a battery."
    return f"Battery is at {int(b.percent)} percent and {'plugged in' if b.power_plugged else 'running on battery'}."


def system_report():
    parts = []
    try:
        import psutil
        parts.append(f"CPU is at {psutil.cpu_percent(interval=0.5):.0f} percent")
        vm = psutil.virtual_memory()
        parts.append(f"RAM {vm.used / 1e9:.1f} of {vm.total / 1e9:.1f} gigabytes")
    except Exception:
        parts.append("I need the psutil package for CPU and RAM")
    if shutil.which("nvidia-smi"):
        r = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"])
        if r and r.returncode == 0 and r.stdout.strip():
            used, total, temp = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            parts.append(f"GPU memory {used} of {total} megabytes at {temp} degrees")
    return ". ".join(parts) + "."


def handle_system_command(window, norm):
    """Short voice commands only (<= 6 words), so normal sentences that merely
    mention 'volume' or 'battery' still go to the chat model."""
    words = norm.split()
    if len(words) > 6:
        return False
    tokens = set(words)
    number = parse_number(words)
    up = bool(tokens & {"up", "increase", "raise", "higher", "louder", "brighter"})
    down = bool(tokens & {"down", "decrease", "lower", "reduce", "quieter", "dimmer", "dim"})

    if tokens & {"mute", "unmute"}:
        mute = "unmute" not in tokens
        ok = set_mute(mute)
        respond(window, ("Muted." if mute else "Unmuted.") if ok else "I couldn't change the mute.", "Standby")
        return True

    if tokens & {"volume", "louder", "quieter"}:
        cur = get_volume()
        if cur is None:
            respond(window, "I couldn't reach the audio system.", "Standby")
        elif number is not None:
            new = set_volume(number)
            respond(window, f"Volume {new} percent." if new is not None else "I couldn't set the volume.", "Standby")
        elif up or down:
            new = set_volume(cur + (10 if up else -10))
            respond(window, f"Volume {new} percent." if new is not None else "I couldn't set the volume.", "Standby")
        else:
            respond(window, f"Volume is {cur} percent.", "Standby")
        return True

    if tokens & {"brightness", "brighter", "dimmer", "dim"}:
        cur = get_brightness()
        if cur is None:
            respond(window, "Brightness control needs the brightnessctl package.", "Standby")
        elif number is not None:
            new = set_brightness(number)
            respond(window, f"Brightness {new} percent." if new is not None else "I couldn't set the brightness.", "Standby")
        elif up or down:
            new = set_brightness(cur + (10 if up else -10))
            respond(window, f"Brightness {new} percent." if new is not None else "I couldn't set the brightness.", "Standby")
        else:
            respond(window, f"Brightness is {cur} percent.", "Standby")
        return True

    if "battery" in tokens:
        respond(window, battery_report(), "Standby")
        return True

    if tokens & {"cpu", "ram", "gpu", "vram"} or {"system", "status"} <= tokens:
        respond(window, system_report(), "Standby")
        return True

    return False


QUESTION_WORDS = {"what", "whats", "how", "why", "when", "who", "where", "which", "is", "does",
                  "do", "did", "are"}


def wants_play(norm):
    """'open the youtube and play tamil songs' -> True.
    'how do I play the guitar' / 'what song should I play' -> False (that's a question)."""
    words = norm.split()
    if "play" not in words or words[0] in QUESTION_WORDS:
        return False
    return "play" in words[:5] or bool({"youtube", "song", "songs", "video", "videos", "music"} & set(words))


SENTENCE_END = re.compile(r"(.+?[.!?\u2026])\s+", re.S)

FAST_REPLIES = {
    "hello": "Hello, sir. How can I help?",
    "hi": "Hi there. What can I do for you?",
    "hey": "Hey. Ready when you are.",
    "greetings": "Greetings, sir. Standing by.",
    "good morning": "Good morning. A fresh start, sir.",
    "good afternoon": "Good afternoon. How may I assist?",
    "good evening": "Good evening. What's on the agenda?",
    "good night": "Good night, sir. Rest well.",
    "thanks": "Always a pleasure, sir.",
    "thank you": "You're most welcome, sir.",
    "thank you jarvis": "Anytime, sir.",
    "thanks jarvis": "At your service.",
    "ok": "Understood.",
    "okay": "Very well.",
    "alright": "Consider it done.",
    "yes": "Affirmative.",
    "no": "Understood.",
    "nice": "Glad you think so.",
    "cool": "Indeed.",
    "wow": "Impressive, isn't it?",
    "sorry": "No need, sir.",
    "bye": "Standing by. Say Hey Jarvis when you need me.",
    "goodbye": "Until next time, sir.",
    "good jarvis": "Thank you, sir. Back to work.",
    "well done": "Thank you, sir. Ready for the next one.",
    "good job": "Appreciate that, sir.",
    "who are you": "I am JARVIS, your personal AI assistant.",
    "what are you": "I'm JARVIS, your desktop companion. I can open apps, play media, change settings, and answer questions.",
    "whats your name": "My name is JARVIS.",
    "your name": "I'm JARVIS.",
    "how are you": "Running at peak efficiency, sir. And you?",
    "how do you do": "Functioning flawlessly. Thank you for asking.",
    # Time/date entries are lambdas so they return the CURRENT time on every call,
    # not the time when the module was first imported.
    "what time is it": lambda: f"It's currently {datetime.datetime.now():%H:%M}.",
    "time": lambda: f"It's {datetime.datetime.now():%H:%M}.",
    "what is the time": lambda: f"The time is {datetime.datetime.now():%H:%M}.",
    "what day is it": lambda: f"Today is {datetime.datetime.now():%A, %d %B %Y}.",
    "whats the date": lambda: f"Today is {datetime.datetime.now():%A %d %B %Y}.",
    "today date": lambda: f"{datetime.datetime.now():%d %B %Y}.",
}


def fast_reply(norm):
    """Instant replies for greetings, pleasantries, time/date — no LLM round-trip.
    Values may be plain strings or zero-argument callables (lambdas) for dynamic content."""
    def _resolve(val):
        return val() if callable(val) else val

    if not norm:
        return None
    if norm in FAST_REPLIES:
        return _resolve(FAST_REPLIES[norm])
    stripped = re.sub(r"[.!?]+$", "", norm).strip()
    if stripped in FAST_REPLIES:
        return _resolve(FAST_REPLIES[stripped])
    for key, val in FAST_REPLIES.items():
        if norm == key or norm.startswith(key + " ") or norm.endswith(" " + key):
            return _resolve(val)
    return None


def chat_streaming(window, *, fast_mode=False):
    """Stream the reply from Ollama; the text grows live in the GUI and each
    finished sentence is spoken immediately instead of waiting for the whole answer."""
    full, buf, speaking = "", "", False
    base = 140
    num_predict = personality_tokens(base, fast_mode)
    timeout = 22 if fast_mode else 45
    # lower temperature = steadier, more reliable answers (benchmarked: default
    # num_ctx stays — shrinking it makes first-token latency worse, not better)
    temp = 0.5 if fast_mode else (0.7 if current_personality() == "buddy" else 0.6)
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "messages": chat_memory, "stream": True,
                  "keep_alive": "30m",
                  "options": {"num_predict": num_predict, "temperature": temp}},
            stream=True, timeout=timeout,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    full += piece
                    buf += piece
                    ui_stream(window, full.strip())
                    if not speaking:
                        ui_status(window, "Speaking...")
                        speaking = True
                    while True:
                        m = SENTENCE_END.match(buf)
                        if not m:
                            break
                        speak(m.group(1))
                        buf = buf[m.end():]
                        if fast_mode and len(full) > 80:
                            break
                if data.get("done") or (fast_mode and len(full) > 140):
                    break
        if buf.strip() and not (fast_mode and len(full) > 140):
            speak(buf)
        print(f"JARVIS: {full.strip()}")
        return full.strip()
    finally:
        ui_stream_end(window)


# ====================================================================== state ====
WINDOW = None            # set in __main__; used by the GUI bridge and the mic-level feed
_cmd_lock = threading.RLock()
_current_command = ""    # what the user just asked (logged together with what JARVIS did)

JARVIS_DIR = os.path.expanduser("~/.jarvis")
CONFIG_PATH = os.path.join(JARVIS_DIR, "config.json")
PERMS_PATH = os.path.join(JARVIS_DIR, "permissions.json")
DB_PATH = os.path.join(JARVIS_DIR, "memory.db")
HOME = os.path.expanduser("~")
SUMMARY_EVERY = 16       # summarise older conversation once this many new turns pile up

def base_prompt_text():
    """Tone instructions pulled from the currently active personality mode."""
    mode = PERSONALITY_MODES.get(current_personality(), PERSONALITY_MODES["jarvis"])
    return (
        mode["tone"] + " "
        "You have a persistent memory: you remember earlier conversations and the actions you performed for the user "
        "(see below). Never say that you have no memory, that a conversation limit was reached, or suggest starting a new chat."
    )


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ================================================================ GUI bridge ====
def ui_eval(js):
    if WINDOW is None:
        return
    try:
        WINDOW.evaluate_js(js)
    except Exception as e:
        print(f"UI error: {e}")


_last_level_push = 0.0


def ui_level_push(rms):
    """Mic loudness -> the orb reacts while you speak."""
    global _last_level_push
    now = time.time()
    if rms and now - _last_level_push < 0.06:
        return
    _last_level_push = now
    ui_eval(f"setLevel({min(1.0, rms / 3000.0):.3f});")


# ======================================================================= memory ====
_db = None
_db_lock = threading.RLock()
_summarizing = False


def mem_init():
    global _db
    os.makedirs(JARVIS_DIR, exist_ok=True)
    _db = sqlite3.connect(DB_PATH, check_same_thread=False)
    with _db_lock:
        _db.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, content TEXT);
            CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, command TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, text TEXT);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        _db.commit()


def _q(sql, args=(), fetch=False):
    if _db is None:
        return []
    with _db_lock:
        cur = _db.execute(sql, args)
        rows = cur.fetchall() if fetch else []
        _db.commit()
    return rows


def mem_add_turn(role, content):
    _q("INSERT INTO turns(ts, role, content) VALUES (?,?,?)", (time.time(), role, content))


def mem_log_action(result):
    if re.match(r"(I couldn't|I can't|I don't|I won't|Which|Nothing|What |Okay, I|You asked me|Volume is|Brightness is|"
                r"Battery is|CPU is|There are \d+ items|I found \d+|I need|Terminal access|Today,|Yesterday,|Recently,)", result):
        return  # failures, questions and read-outs aren't "things I did"
    _q("INSERT INTO actions(ts, command, result) VALUES (?,?,?)", (time.time(), _current_command, result))


def mem_meta_get(key, default=""):
    rows = _q("SELECT value FROM meta WHERE key=?", (key,), fetch=True)
    return rows[0][0] if rows else default


def mem_meta_set(key, value):
    _q("INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def mem_facts():
    return [r[0] for r in _q("SELECT text FROM facts ORDER BY id", fetch=True)]


def mem_add_fact(text):
    _q("INSERT INTO facts(ts, text) VALUES (?,?)", (time.time(), text))


def mem_count():
    t = _q("SELECT COUNT(*) FROM turns", fetch=True)
    a = _q("SELECT COUNT(*) FROM actions", fetch=True)
    return (t[0][0] if t else 0) + (a[0][0] if a else 0)


def mem_clear_all():
    for table in ("turns", "actions", "facts", "meta"):
        _q(f"DELETE FROM {table}")
    chat_memory[:] = [{"role": "system", "content": build_system_prompt()}]


def mem_actions(since_ts, until_ts=None, limit=12):
    """Newest `limit` actions in the range, returned oldest -> newest as (ts, command, result)."""
    until_ts = until_ts or time.time() + 1
    rows = _q("SELECT ts, command, result FROM actions WHERE ts>=? AND ts<? ORDER BY id DESC LIMIT ?",
              (since_ts, until_ts, limit), fetch=True)
    return list(reversed(rows))


def mem_recent_turns(n=6):
    rows = _q("SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (n,), fetch=True)
    turns = [{"role": r, "content": c} for r, c in reversed(rows)]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def build_system_prompt():
    parts = [base_prompt_text(), f"Current date and time: {datetime.datetime.now():%A %d %B %Y, %H:%M}."]
    facts = mem_facts()
    if facts:
        parts.append("Things the user asked you to remember: " + "; ".join(facts[-15:]) + ".")
    summary = mem_meta_get("summary")
    if summary:
        parts.append("Summary of your earlier conversations with the user: " + summary)
    recent = mem_actions(time.time() - 86400, limit=6)
    if recent:
        items = [f"{datetime.datetime.fromtimestamp(ts):%H:%M} {res}" for ts, _cmd, res in recent]
        parts.append("Actions you recently performed for the user (oldest first): " + " | ".join(items))
    return "\n".join(parts)


def refresh_system_prompt():
    chat_memory[0]["content"] = build_system_prompt()


def bootstrap_memory():
    """On startup: reload what happened before, so JARVIS continues instead of starting blank."""
    mem_init()
    chat_memory[:] = [{"role": "system", "content": build_system_prompt()}] + mem_recent_turns(6)


def push_memory():
    ui_eval(f"setMemory({mem_count()});")


def maybe_summarize():
    """Old conversation is compressed into a running summary instead of being dropped -> no 'limit reached'."""
    global _summarizing
    if _summarizing:
        return
    last = int(mem_meta_get("summarized_upto", "0") or 0)
    rows = _q("SELECT COUNT(*), MAX(id) FROM turns WHERE id>?", (last,), fetch=True)
    count, max_id = rows[0] if rows else (0, 0)
    if (count or 0) < SUMMARY_EVERY:
        return
    _summarizing = True
    threading.Thread(target=_summarize_worker, args=(last, max_id - 6), daemon=True).start()


def _summarize_worker(last, cutoff):
    global _summarizing
    try:
        rows = _q("SELECT role, content FROM turns WHERE id>? AND id<=? ORDER BY id", (last, cutoff), fetch=True)
        if not rows:
            return
        convo = "\n".join(f"{'User' if r == 'user' else 'JARVIS'}: {c}" for r, c in rows)
        prompt = ("Update this running summary of conversations between a user and their assistant JARVIS. "
                  "Keep names, preferences, facts about the user and tasks that were done. Max 120 words, plain text.\n\n"
                  f"Previous summary: {mem_meta_get('summary') or '(none)'}\n\nNew conversation:\n{convo}\n\nUpdated summary:")
        r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "stream": False, "keep_alive": "30m",
                                            "messages": [{"role": "user", "content": prompt}],
                                            "options": {"num_predict": 220}}, timeout=180)
        text = r.json()["message"]["content"].strip()
        if text:
            mem_meta_set("summary", text[:1200])
            mem_meta_set("summarized_upto", str(cutoff))
    except Exception as e:
        print(f"Summary error: {e}")
    finally:
        _summarizing = False


def _day_label(ts):
    d = datetime.datetime.fromtimestamp(ts).date()
    today = datetime.date.today()
    if d == today:
        return "today"
    if d == today - datetime.timedelta(days=1):
        return "yesterday"
    return f"{d:%a %d %b}"


def handle_memory_command(window, norm, command):
    words = norm.split()
    tokens = set(words)

    # "remember that my exam is on Friday"
    m = re.match(r"^\W*(?:hey jarvis\W+)?(?:please\s+)?remember\s+(?:that\s+)?(.+?)\W*$", command, re.I | re.S)
    if m and len(m.group(1)) >= 3:
        fact = m.group(1).strip()
        mem_add_fact(fact)
        respond(window, f"Okay, I'll remember that {fact}.", "Standby")
        return True

    # "what do you remember / know about me"
    if "what" in tokens and "you" in tokens and ({"remember", "know"} & tokens) and ({"me", "about"} & tokens):
        facts = mem_facts()
        summary = mem_meta_get("summary")
        if facts:
            respond(window, "You asked me to remember: " + "; ".join(facts[-8:]) + ".", "Standby")
        elif summary:
            respond(window, summary, "Standby")
        else:
            respond(window, "You haven't asked me to remember anything yet.", "Standby")
        return True

    # "forget everything"
    if "forget" in tokens and ({"everything", "all", "memory", "memories"} & tokens):
        if ask_permission(window, "ERASE JARVIS MEMORY?",
                          "This permanently deletes remembered facts, conversation history, summaries and the action log."):
            mem_clear_all()
            respond(window, "Memory erased. We start fresh.", "Standby")
        else:
            respond(window, "Okay, I kept my memory.", "Standby")
        return True

    # "what did you do today / yesterday / before", "show history"
    asks_done = ({"what", "show", "list"} & tokens) and (
        (({"did", "have", "had"} & tokens) and ({"you", "we"} & tokens)
         and ({"do", "done", "open", "opened", "play", "played", "run", "ran"} & tokens))
        or ({"history", "log", "activity"} & tokens))
    asks_said = ({"what", "show"} & tokens) and ({"did", "have"} & tokens) and "i" in tokens \
        and ({"ask", "asked", "say", "said"} & tokens)
    if asks_done or asks_said:
        now = datetime.datetime.now()
        midnight = datetime.datetime.combine(now.date(), datetime.time.min).timestamp()
        if "yesterday" in tokens:
            rows, label = mem_actions(midnight - 86400, midnight, 15), "yesterday"
        elif "today" in tokens:
            rows, label = mem_actions(midnight, None, 15), "today"
        else:
            rows, label = mem_actions(0, None, 10), "recently"
        if not rows:
            respond(window, f"I haven't done anything {label} yet.", "Standby")
            return True
        lines = [f"{_day_label(ts)} {datetime.datetime.fromtimestamp(ts):%H:%M}  -  "
                 f"{('you asked: ' + cmd) if asks_said else res}" for ts, cmd, res in rows]
        ui_message(window, "jarvis", "\n".join(lines))
        newest = [(cmd if asks_said else res) for _ts, cmd, res in rows[-3:]]
        what = "you asked" if asks_said else "I did"
        respond(window, f"{label.capitalize()}, {what} {len(rows)} things. Most recent: " + "; ".join(reversed(newest)), "Standby", log=False)
        return True
    return False


# ========================================================================= colors ====
COLORS = {
    "gold": (36, 100), "orange": (22, 100), "red": (0, 100), "pink": (330, 90), "purple": (275, 85),
    "blue": (212, 100), "cyan": (186, 100), "teal": (165, 90), "green": (130, 90), "yellow": (55, 100),
    "white": (0, 0),
}
COLOR_ALIASES = {"golden": "gold", "amber": "gold", "default": "gold", "original": "gold", "normal": "gold",
                 "violet": "purple", "magenta": "pink", "aqua": "cyan", "grey": "white", "gray": "white"}


def apply_theme(name, persist=True, push=True):
    h, s = COLORS[name]
    if push:
        ui_eval(f"setThemeHSL({json.dumps(name)}, {h}, {s});")
    if persist:
        cfg = _read_json(CONFIG_PATH, {})
        cfg["theme"] = name
        _write_json(CONFIG_PATH, cfg)


def handle_voice_command(window, norm, command):
    """Change TTS voice (male/female/ryan/lessac/joe/libritts) and voice downloads."""
    words = norm.split()
    tokens = set(words)

    also_alias = any(w in VOICE_MODELS or w in VOICE_ALIASES for w in words)
    voice_words = {"voice", "voices", "accent", "speak", "speech", "sound", "talk", "tone", "gender"}
    has_voice_word = bool(voice_words & tokens)
    verbs = {"change", "switch", "use", "turn", "set", "make", "give", "pick", "go"}
    has_verb = bool(verbs & set(words[:5]))

    if not has_voice_word and not also_alias:
        return False
    if has_voice_word or has_verb:
        pass
    elif len(words) == 1 and also_alias:
        pass
    else:
        return False

    download = bool(tokens & {"download", "fetch", "get", "install", "grab"})
    list_request = bool(tokens & {"list", "what", "which", "show", "available"}) and has_voice_word

    if list_request and not also_alias:
        lines = []
        for n, g, label, avail in voice_list():
            tag = "ready" if avail else "not downloaded"
            lines.append(f"[{tag}] {label}")
        ui_message(window, "jarvis", "\n".join(lines))
        ready_names = [n for n, _g, _l, avail in voice_list() if avail]
        total = len(VOICE_MODELS)
        respond(window, f"I have {len(ready_names)} of {total} voices ready: " + ", ".join(ready_names) +
                f". {total - len(ready_names)} more can be downloaded. Say: download voice ryan, or change voice to male.",
                "Standby", log=False)
        return True

    requested = None
    for w in words:
        if w in VOICE_MODELS:
            requested = w
        elif w in VOICE_ALIASES:
            requested = VOICE_ALIASES[w]
    if requested is None:
        if {"man", "male", "guy", "boy"} & tokens:
            requested = "ryan"
        elif {"woman", "female", "lady", "girl"} & tokens:
            requested = "lessac"

    if not requested:
        if download:
            respond(window, "Which voice? I can download ryan male, joe male, lessac female, or libritts female.", "Listening...")
        else:
            respond(window, "Which voice? Try male (ryan), female (lessac), joe, libritts, or say list voices.", "Listening...")
        return True

    auto_download = True
    ok, msg = set_voice(requested, download_if_missing=auto_download, window=window)
    respond(window, msg, "Standby")
    return True


def handle_personality_command(window, norm):
    """buddy / jarvis mode toggle + companion switch."""
    words = norm.split()
    tokens = set(words)
    joined = " " + norm + " "

    buddy_tokens = {"buddy", "friend", "pal", "companion", "chill", "casual", "friendly", "mate"}
    jarvis_tokens = {"jarvis", "pro", "professional", "assistant", "formal", "work", "butler", "tony", "original", "normal", "smart"}

    # "jarvis" / "friend" alone is praise or small talk ("Good Jarvis?"), not a
    # command — only switch when there is an imperative or a mode word.
    imperative = bool({"be", "act", "switch", "change", "turn", "set", "go", "become",
                       "stay", "keep", "sound", "talk", "mode"} & tokens)
    mode = None
    if buddy_tokens & tokens and (imperative or len(words) == 1):
        mode = "buddy"
    elif jarvis_tokens & tokens and (imperative or len(words) == 1):
        mode = "jarvis"

    be_my_phrases = ["be my buddy", "be my friend", "be my pal", "be my companion",
                     "act like a buddy", "be more friendly", "be a friend", "be chill",
                     "talk like a buddy", "be casual", "chat with me"]
    for p in be_my_phrases:
        if p in joined or p in norm:
            mode = "buddy"
    be_pro_phrases = ["be professional", "be formal", "be jarvis", "be the butler",
                      "switch back to jarvis", "act like jarvis", "be an assistant"]
    for p in be_pro_phrases:
        if p in joined or p in norm:
            mode = "jarvis"

    if mode is None:
        return False
    ok, msg = set_personality(mode)
    refresh_system_prompt()
    if mode == "buddy":
        respond(window, msg + " Now let's chat — what's up?", "Standby")
    else:
        respond(window, msg + " Back to work, sir.", "Standby")
    return True


def handle_theme_command(window, norm):
    words = norm.split()
    tokens = set(words)

    names = []
    for w in words:
        if w in COLORS:
            names.append(w)
        elif w in COLOR_ALIASES:
            names.append(COLOR_ALIASES[w])

    if len(words) == 1 and names:
        apply_theme(names[0])
        respond(window, f"Color changed to {names[0]}.", "Standby")
        return True

    if len(words) == 2 and names and (tokens & {"color", "colour", "theme", "please"}):
        apply_theme(names[0])
        respond(window, f"Color changed to {names[0]}.", "Standby")
        return True

    verbs = {"change", "set", "make", "turn", "switch", "use", "paint", "go", "give", "apply", "pick", "choose", "select"}
    target = tokens & {"it", "ui", "interface", "screen", "color", "colour", "theme", "jarvis", "orb", "everything", "display",
                       "that", "this", "the", "to"}

    verb_ok = bool(verbs & set(words[:6]))
    color_mention = bool(tokens & {"color", "colour", "colors", "colours", "theme"})

    if names and (verb_ok or color_mention or len(words) <= 4):
        name = names[-1]
        apply_theme(name)
        respond(window, f"Color changed to {name}.", "Standby")
        return True

    if verb_ok and (color_mention or target) and not names:
        respond(window, "Which color? I can do " + ", ".join(COLORS) + ".", "Listening...")
        return True

    return False


# ===================================================================== permissions ====
_perm_pending = {}
FOLDER_MAP = {"documents": "Documents", "document": "Documents", "downloads": "Downloads", "download": "Downloads",
              "desktop": "Desktop", "pictures": "Pictures", "photos": "Pictures", "music": "Music", "videos": "Videos"}
_XDG = {"Documents": "DOCUMENTS", "Downloads": "DOWNLOAD", "Desktop": "DESKTOP",
        "Pictures": "PICTURES", "Music": "MUSIC", "Videos": "VIDEOS"}
BARE_FOLDER_WORDS = {"documents", "document", "downloads", "download", "desktop", "pictures", "photos"}

# Never run, even with "full access" and even if you click Allow.
BLOCKED_COMMANDS = [
    r"\bsudo\b", r"\bsu\s", r"\bpkexec\b", r"\bmkfs", r"\bdd\s+if=", r":\(\)\s*\{", r"\b(shutdown|reboot|poweroff|halt)\b",
    r">\s*/dev/sd", r"chmod\s+-R\s+7?77\s+/", r"curl[^|]*\|\s*(ba)?sh", r"wget[^|]*\|\s*(ba)?sh", r"/etc/(passwd|shadow|sudoers)",
    r"\brm\s+(-\w*\s+)*-\w*[rf]\w*\s+(-\w+\s+)*(/|~|\$HOME|\*)(\s|$)",
]


def load_perms():
    p = _read_json(PERMS_PATH, {})
    return {"folders": list(p.get("folders", [])), "shell": bool(p.get("shell", False))}


def save_perms(p):
    _write_json(PERMS_PATH, p)
    push_access()


def access_level():
    p = load_perms()
    return "FULL" if p["shell"] else "FILES" if p["folders"] else "LOCKED"


def push_access():
    ui_eval(f"setAccess({json.dumps(access_level())});")


def user_folder(name):
    if name in _XDG and shutil.which("xdg-user-dir"):
        r = run(["xdg-user-dir", _XDG[name]])
        if r and r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return os.path.join(HOME, name)


def path_ok(path):
    """Inside a granted folder, and not a hidden file/folder (so .ssh, .gnupg, browser profiles... stay private)."""
    rp = os.path.realpath(path)
    if rp == HOME or rp.startswith(HOME + os.sep):
        if any(part.startswith(".") and part != "." for part in os.path.relpath(rp, HOME).split(os.sep)):
            return False
    return any(rp == root or rp.startswith(root.rstrip(os.sep) + os.sep) for root in load_perms()["folders"])


def ask_permission(window, title, detail, timeout=45):
    """Shows Allow / Deny on screen. Only a click counts - never a spoken 'yes', which noise or a video could fake."""
    req_id = str(time.time_ns())
    ev = threading.Event()
    _perm_pending[req_id] = {"event": ev, "allowed": False}
    ui_eval(f"showPermission({json.dumps(req_id)}, {json.dumps(title)}, {json.dumps(detail)});")
    ui_status(window, "Awaiting permission...")
    speak("I need your permission. Please confirm on the screen.")
    ev.wait(timeout)
    allowed = _perm_pending.pop(req_id, {}).get("allowed", False)
    ui_eval("hidePermission();")
    return allowed


def handle_access_command(window, norm):
    words = norm.split()
    tokens = set(words)
    first = set(words[:4])
    joined = " " + norm + " "

    access_words = tokens & {"access", "permission", "permissions", "control", "power", "authority"}
    full_words = {"full", "everything", "complete", "unrestricted", "all", "total", "entire", "whole", "everything"}
    unlock_words = {"unlock", "grant", "give", "allow", "enable", "let", "turn on", "activate", "open up"}
    lock_words = {"lock", "revoke", "remove", "withdraw", "disable", "reset", "deny", "take away", "close"}

    # Lock / revoke / lockdown
    if (lock_words & first and access_words) or \
       norm in {"lock down", "lockdown", "lock everything", "lock it down", "lock me down",
                "revoke all access", "disable everything", "remove all permissions"} or \
       (("lock down" in joined or "lock it down" in joined or "revoke all" in joined or
         "disable all" in joined or "take everything" in joined) and
        (access_words or {"system", "control", "power"} & tokens)):
        save_perms({"folders": [], "shell": False})
        respond(window, "All access revoked. I'm locked down.", "Standby")
        return True

    # "what access / permissions / status do you have" — status
    if (access_words and ({"what", "which", "show", "check", "status", "current"} & tokens)
            and not ((unlock_words | {"give", "grant"}) & first)):
        p = load_perms()
        if p["shell"]:
            msg = "I have full access: files in your home folder, and terminal commands that you approve one by one."
        elif p["folders"]:
            msg = "I can use these folders: " + ", ".join(os.path.basename(f) or "home" for f in p["folders"]) + "."
        else:
            msg = "I'm locked down. I can only open apps, play media and change settings."
        respond(window, msg, "Standby", log=False)
        return True

    # Grant / unlock / give access
    wants_grant = bool((unlock_words | {"give", "grant"}) & first) or \
                  any(p in joined for p in
                      ["unlock full", "grant full", "give me full", "enable all", "give all",
                       "unlock everything", "grant everything", "give me everything",
                       "let me control all", "full control", "full system", "full permissions",
                       "complete control", "total access", "all permissions", "all access",
                       "unlock all", "open everything"])
    wants_full = bool((full_words | {"system", "control"}) & tokens) or \
                 any(p in joined for p in
                     ["everything", "full access", "full system", "full control", "unlock all",
                      "complete access", "total control", "all the control", "all the access"])

    if wants_grant or (access_words and wants_full):
        full = wants_full
        folders = []
        for w in words:
            if w in FOLDER_MAP:
                folders.append(user_folder(FOLDER_MAP[w]))
        if "home" in tokens:
            folders.append(HOME)
        if full:
            folders = [HOME]
        folders = list(dict.fromkeys(folders))
        if not folders and not full:
            respond(window, "Which folder? Documents, Downloads, Desktop, Pictures, Music, Videos, your home folder, or say: give full access.",
                    "Listening...")
            return True
        if full or not folders:
            # default to home when "unlock full …" was said but no folder named
            folders = [HOME]
            full = True
        if full:
            title = "GRANT FULL ACCESS?"
            detail = (f"JARVIS will be able to find, list and open any visible file in {HOME}, and run terminal commands.\n\n"
                      "Every terminal command still asks you first and shows the exact command. Hidden folders (.ssh, .gnupg, "
                      "browser profiles), sudo, disk formatting, shutdown and deleting system files stay blocked.")
        else:
            title = "GRANT FILE ACCESS?"
            detail = "JARVIS will be able to find, list and open files in:\n" + "\n".join(folders)
        if ask_permission(window, title, detail):
            p = load_perms()
            p["folders"] = list(dict.fromkeys(p["folders"] + folders))
            p["shell"] = p["shell"] or full
            save_perms(p)
            respond(window, "Full access granted. You can now use the full system via approved commands and home folder files."
                    if full else "Access granted.", "Standby")
        else:
            respond(window, "Okay, I won't get access.", "Standby")
        return True

    return False


# ============================================================== files & terminal ====
EXT_WORDS = {"pdf", "doc", "docx", "txt", "ppt", "pptx", "xlsx", "csv", "png", "jpg", "jpeg", "mp3", "mp4", "zip", "py"}
FILE_STOP = {"find", "search", "locate", "look", "for", "the", "my", "a", "an", "file", "files", "called", "named", "open",
             "please", "in", "on", "inside", "show", "list", "me", "folder", "directory", "from", "launch", "home", "to", "of"}
SKIP_DIRS = {"node_modules", "venv", ".venv", "__pycache__", "site-packages"}


def search_files(roots, name_tokens, ext, limit=5, max_scan=40000):
    found, scanned = [], 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
            for fn in filenames:
                scanned += 1
                if scanned > max_scan:
                    break
                if fn.startswith("."):
                    continue
                low = re.sub(r"[^a-z0-9]+", " ", fn.lower())
                if ext and not fn.lower().endswith("." + ext):
                    continue
                if all(t in low for t in name_tokens):
                    full = os.path.join(dirpath, fn)
                    try:
                        found.append((os.path.getmtime(full), full))
                    except OSError:
                        pass
            if scanned > max_scan:
                break
    found.sort(reverse=True)
    return [p for _m, p in found[:limit]], len(found)


def _xdg_open(path):
    subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def handle_file_command(window, norm):
    words = norm.split()
    tokens = set(words)
    verb = next((w for w in words[:4] if w in {"open", "find", "search", "locate", "list", "show", "launch"}), None)
    if not verb:
        return False
    folder_word = next((w for w in words if w in FOLDER_MAP), None)
    if folder_word is None and "home" in tokens:
        folder_word = "home"
    ext = next((w for w in words if w in EXT_WORDS), None)
    file_intent = bool(tokens & {"file", "files"}) or ext is not None
    if tokens & {"manager", "explorer"}:                       # "open file manager" is an app, not a file search
        return False

    folder_path = None
    if folder_word:
        folder_path = HOME if folder_word == "home" else user_folder(FOLDER_MAP[folder_word])
    name_tokens = [w for w in words if w not in FILE_STOP and w not in FOLDER_MAP and w not in EXT_WORDS]

    # --- open a folder in the file manager
    if verb in {"open", "show", "launch"} and folder_word and not file_intent \
            and (folder_word in BARE_FOLDER_WORDS or tokens & {"folder", "directory"}):
        if not path_ok(folder_path):
            respond(window, f"I don't have permission for {folder_word} yet. Say: give access to {folder_word}.", "Standby")
            return True
        _xdg_open(folder_path)
        respond(window, f"Opening your {folder_word} folder.", "Launching App...")
        return True

    # --- list files in a folder
    if verb in {"list", "show"} and folder_word and file_intent:
        if not path_ok(folder_path):
            respond(window, f"I don't have permission for {folder_word} yet. Say: give access to {folder_word}.", "Standby")
            return True
        items = [n for n in os.listdir(folder_path) if not n.startswith(".")]
        items.sort(key=lambda n: os.path.getmtime(os.path.join(folder_path, n)), reverse=True)
        ui_message(window, "jarvis", f"{folder_word}:\n" + "\n".join(items[:15]))
        respond(window, f"There are {len(items)} items in {folder_word}. The newest are: " + ", ".join(items[:3]) + ".",
                "Standby", log=False)
        return True

    # --- find / open a file by name
    if (verb in {"find", "search", "locate"} and file_intent) or (verb == "open" and file_intent):
        opening = verb == "open"
        if opening and not name_tokens and not ext:
            return False                          # plain "open files" -> let the app launcher handle it
        if not name_tokens and not ext:
            respond(window, "What file should I look for?", "Listening...")
            return True
        roots = [folder_path] if folder_path else load_perms()["folders"]
        roots = [r for r in roots if r and path_ok(r)]
        if not roots:
            respond(window, "I need permission first. Say: give access to documents, or give full access.", "Standby")
            return True
        results, total = search_files(roots, name_tokens, ext)
        if not results:
            respond(window, "I couldn't find a matching file in the folders I can access.", "Standby", log=False)
            return True
        ui_message(window, "jarvis", "\n".join(results))
        if opening:
            _xdg_open(results[0])
            respond(window, f"Opening {os.path.basename(results[0])}.", "Launching App...")
        else:
            respond(window, f"I found {total} matching files. The newest is {os.path.basename(results[0])} in "
                            f"{os.path.basename(os.path.dirname(results[0]))}.", "Standby", log=False)
        return True
    return False


SHELL_RE = re.compile(
    r"^\W*(?:hey jarvis\W+)?(?:please\s+)?(?:(?:run|execute)\s+(?:the\s+|a\s+)?(?:terminal\s+|shell\s+)?command"
    r"|terminal\s+command|shell\s+command)\s*[:,]?\s+(.+?)\s*$", re.I | re.S)
TYPED_RE = re.compile(r"^\s*[$!]\s+(.+?)\s*$", re.S)


def handle_shell_command(window, command):
    m = SHELL_RE.match(command)
    voice = bool(m)
    m = m or TYPED_RE.match(command)
    if not m:
        return False
    cmd = m.group(1).strip()
    if voice:
        cmd = re.sub(r"(?<=\w)\.$", "", cmd)     # Whisper adds a full stop at the end of sentences
    if not load_perms()["shell"]:
        respond(window, "Terminal access is locked. Say: give full access, and confirm on the screen.", "Standby")
        return True
    if any(re.search(p, cmd, re.I) for p in BLOCKED_COMMANDS):
        respond(window, "I won't run that. It's on my blocked list: sudo, disk formatting, shutdown, or deleting system files.",
                "Standby")
        return True
    if not ask_permission(window, "RUN THIS COMMAND?", cmd):
        respond(window, "Okay, I didn't run it.", "Standby")
        return True
    ui_status(window, "Working...")
    try:
        r = subprocess.run(cmd, shell=True, cwd=HOME, capture_output=True, text=True, timeout=30, executable="/bin/bash")
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        code = r.returncode
    except subprocess.TimeoutExpired:
        out, code = "(stopped after 30 seconds)", 124
    ui_message(window, "jarvis", f"$ {cmd}\n{out[:800] or '(no output)'}")
    mem_log_action(f"Ran the terminal command {cmd} (exit code {code})")
    first = out.splitlines()[0] if out else ""
    if code == 0:
        spoken = "Done." + (f" {first}" if first and len(first) < 100 else "")
    else:
        spoken = f"The command failed with code {code}."
    respond(window, spoken, "Standby", log=False)
    return True


# =================================================================== GUI -> Python ====
class Api:
    """Methods the page can call (window.pywebview.api.*)."""

    def set_theme(self, name):
        if name in COLORS:
            apply_theme(name, persist=True, push=False)

    def cycle_access(self):
        if WINDOW is None:
            return
        threading.Thread(target=_cycle_access_worker, args=(WINDOW,), daemon=True).start()

    def permission_response(self, req_id, allowed):
        item = _perm_pending.get(req_id)
        if item:
            item["allowed"] = bool(allowed)
            item["event"].set()

    def submit_text(self, text):
        text = (text or "").strip()
        if text and WINDOW is not None:
            threading.Thread(target=process_command, args=(text, WINDOW), daemon=True).start()


def _cycle_access_worker(window):
    cur = access_level()
    if cur == "LOCKED":
        folders = [user_folder(FOLDER_MAP[k]) for k in
                   ["documents", "downloads", "desktop", "pictures", "music", "videos"]]
        folders = list(dict.fromkeys([f for f in folders if f]))
        detail = "JARVIS will be able to find, list and open files in:\n" + "\n".join(folders)
        if ask_permission(window, "GRANT FILE ACCESS?", detail):
            p = load_perms()
            p["folders"] = list(dict.fromkeys(p["folders"] + folders))
            save_perms(p)
            respond(window, "File access granted. Documents, Downloads, Desktop, Pictures, Music, Videos.", "Standby")
        else:
            respond(window, "Okay, staying locked.", "Standby")
    elif cur == "FILES":
        title = "GRANT FULL ACCESS?"
        detail = (f"JARVIS will be able to use any visible file in {HOME}, and run terminal commands you approve.\n\n"
                  "Hidden folders (.ssh, .gnupg, browser profiles), sudo, disk formatting, shutdown and deleting "
                  "system files stay blocked.")
        if ask_permission(window, title, detail):
            p = load_perms()
            p["folders"] = list(dict.fromkeys(p["folders"] + [HOME]))
            p["shell"] = True
            save_perms(p)
            respond(window, "Full access granted.", "Standby")
        else:
            respond(window, "Okay, keeping file-only access.", "Standby")
    else:
        save_perms({"folders": [], "shell": False})
        respond(window, "All access revoked. I'm locked down.", "Standby")


def ui_init(window):
    ui_eval(f"initThemes({json.dumps({k: list(v) for k, v in COLORS.items()})});")
    cfg = _read_json(CONFIG_PATH, {})
    name = cfg.get("theme", "gold")
    apply_theme(name if name in COLORS else "gold", persist=False)
    v = cfg.get("voice")
    if v in VOICE_MODELS and voice_available(v):
        set_voice(v, persist=False)
    p = cfg.get("personality")
    if p in PERSONALITY_MODES:
        set_personality(p, persist=False)
    refresh_system_prompt()
    push_access()
    push_memory()


def process_command(command, window):
    """One command at a time (voice and typed input share this lock)."""
    global _current_command
    with _cmd_lock:
        _current_command = command
        try:
            _process_command(command, window)
        finally:
            push_memory()


_COMPOUND_SPLITTERS = re.compile(
    r'\s+(?:and\s+then|then|after\s+that|next|and)\s+',
    re.IGNORECASE
)

# Verbs that are valid as sub-commands so we don't split on innocuous "and"
_COMMAND_STARTERS = {
    "open", "launch", "start", "close", "shut", "type", "search", "google",
    "browse", "play", "pause", "stop", "volume", "brightness", "set",
    "enable", "disable", "show", "hide", "run", "execute", "terminal",
    "find", "where", "what", "who",
}


def _split_compound_command(command):
    """'Open Firefox and type Hello World' → ['Open Firefox', 'type Hello World'].
    Only splits when the second part starts with a known command verb so we
    don't break normal sentences that happen to contain 'and'.
    """
    parts = _COMPOUND_SPLITTERS.split(command.strip())
    if len(parts) < 2:
        return [command]
    result = [parts[0].strip()]
    for part in parts[1:]:
        first_word = normalize(part).split()[0] if part.strip() else ""
        if first_word in _COMMAND_STARTERS:
            result.append(part.strip())
        else:
            # Not a command verb — re-attach to the previous part
            result[-1] = result[-1] + " " + part.strip()
    return [p for p in result if p]


def _process_command(command, window):
    # ── Compound command handling ────────────────────────────────────────────
    # "Open Firefox and type Hello" → runs each sub-command sequentially.
    sub_cmds = _split_compound_command(command)
    if len(sub_cmds) > 1:
        for sub in sub_cmds:
            _process_command(sub, window)
        return

    norm = normalize(command)
    tokens = set(norm.split())

    # Ignore silence/noise hallucinations
    if len(norm) < 2 or "www" in norm or norm in NOISE_WORDS:
        print(f"Ignored noise/hallucination: {command}")
        return

    # −1) Shut JARVIS down entirely
    if _is_jarvis_exit(norm):
        respond(window, "Shutting down. Goodbye, sir.", "Offline")
        time.sleep(1.2)
        import os as _os
        _os._exit(0)

    ui_message(window, "user", command)

    # 0) typing / dictation plugin (intercepts everything while it is on)
    if handle_typing_command(window, norm, command):
        return

    # 1) say / repeat
    if norm.startswith("say ") or norm.startswith("repeat "):
        phrase = command.split(" ", 1)[1].strip()
        if phrase:
            respond(window, phrase, "Speaking...")
        else:
            respond(window, "What should I say?", "Speaking...")

    # colors, memory, permissions, files, terminal
    elif handle_personality_command(window, norm):
        pass

    elif handle_voice_command(window, norm, command):
        pass

    elif handle_theme_command(window, norm):
        pass

    elif handle_memory_command(window, norm, command):
        pass

    elif handle_access_command(window, norm):
        pass

    elif handle_shell_command(window, command):
        pass

    elif handle_file_command(window, norm):
        pass

    # 2) play something (whole-word match, so "display" / "condition" don't trigger it)
    elif {"stop", "close", "quit"} & tokens and {"music", "song", "songs", "video", "videos", "player"} & tokens:
        stop_player()
        respond(window, "Stopping playback.", "Standby")

    # pause / resume / next / previous (our player, or Spotify/VLC/etc. via playerctl)
    elif handle_media_command(window, norm):
        pass

    elif wants_play(norm):
        song = clean_query(norm)
        if song:
            respond(window, f"Right away. Playing {song}.", "Playing Media...")
            play_on_youtube(song)
        else:
            respond(window, "Opening YouTube.", "Browsing...")
            open_url_browser("https://www.youtube.com")

    # 3) open / search YouTube
    elif "youtube" in tokens and ({"open", "search", "launch"} & tokens):
        query = clean_query(norm)
        if query:
            respond(window, f"Searching YouTube for {query}.", "Browsing...")
            open_url_browser(
                f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")
        else:
            respond(window, "Opening YouTube.", "Browsing...")
            open_url_browser("https://www.youtube.com")

    # 3b) search the web (after YouTube so "search ... on youtube" keeps working)
    elif handle_web_search(window, norm):
        pass

    # volume / brightness / battery / CPU-RAM-GPU status
    elif handle_system_command(window, norm):
        pass

    # 4) open / close ANY installed application (sentence must START with the verb)
    elif handle_app_command(window, norm):
        pass

    # 5) instant reply for greetings, pleasantries, time/date
    else:
        instant = fast_reply(norm)
        if instant:
            respond(window, instant, "Standby")
            chat_memory.append({"role": "user", "content": command})
            chat_memory.append({"role": "assistant", "content": instant})
            mem_add_turn("user", command)
            mem_add_turn("assistant", instant)
            return

        fast_mode = False
        user_query = command
        if {"fast", "quick", "quickly", "now", "hurry", "short"} & tokens:
            fast_mode = True
            stripped = re.sub(r"\b(fast|quick|quickly|now|hurry|short|brief|shortly)\b", "", norm, flags=re.I).strip()
            if len(stripped) >= 2:
                user_query = re.sub(r"\b(fast|quick|quickly|now|hurry|short|brief|shortly)\b", "", command, flags=re.I).strip()

        ui_status(window, "Thinking...")
        refresh_system_prompt()
        chat_memory.append({"role": "user", "content": user_query})
        try:
            reply = chat_streaming(window, fast_mode=fast_mode) or "Done."
            chat_memory.append({"role": "assistant", "content": reply})
            mem_add_turn("user", user_query)
            mem_add_turn("assistant", reply)
            maybe_summarize()
            while len(chat_memory) > 11:
                del chat_memory[1:3]  # drop the oldest user+assistant PAIR, keep the system prompt
        except RuntimeError as e:  # Ollama answered with an error (e.g. model not pulled)
            chat_memory.pop()
            print(f"Ollama error: {e}")
            ui_message(window, "jarvis", f"Brain error: {e}")
            ui_status(window, "Error")
            time.sleep(2)
        except Exception as e:
            chat_memory.pop()  # drop the unanswered user message
            print(f"Connection Error: {e}")
            ui_message(window, "jarvis", "Brain disconnected. Is Ollama running?")
            ui_status(window, "Disconnected")
            time.sleep(2)


# --------------------------------------------------------------- backend ----
def load_whisper():
    """RTX GPU if usable (bigger, more accurate model), otherwise CPU."""
    if WHISPER_DEVICE in ("auto", "cuda"):
        try:
            model = WhisperModel("small", device="cuda", compute_type="float16")
            # warm-up: missing CUDA libraries usually only show up on the first transcribe
            list(model.transcribe(np.zeros(RATE, dtype=np.float32), beam_size=1)[0])
            print("Whisper: GPU (small, float16)")
            return model
        except Exception as e:
            print(f"GPU Whisper unavailable ({e}); falling back to CPU")
    print("Whisper: CPU (base, int8)")
    return WhisperModel("base", device="cpu", compute_type="int8")


def _warm_brain():
    """Pre-load the Ollama model at startup so the first question doesn't pay the
    cold-start cost (benchmarked: 7.7s cold vs 0.55s warm time-to-first-token)."""
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "keep_alive": "30m", "stream": False,
                  "messages": [{"role": "user", "content": "hello"}]},
            timeout=120,
        )
        print(f"Brain warm ({OLLAMA_MODEL}) ready")
    except Exception as e:
        print(f"Brain warm-up failed: {e}")


def load_wake_model():
    def find():
        return [m for m in openwakeword.get_pretrained_model_paths() if "hey_jarvis" in m.lower()]

    paths = find()
    if not paths:
        print("Wake-word model not found locally; downloading...")
        openwakeword.utils.download_models()
        paths = find()
    if not paths:
        raise RuntimeError("Could not locate the 'hey_jarvis' wake-word model.")

    try:  # newer openwakeword
        return Model(wakeword_models=paths)
    except TypeError:  # older openwakeword
        return Model(wakeword_model_paths=paths)


def run_jarvis_backend(window):
    # Don't touch the UI until the page has finished loading
    window.events.loaded.wait()
    ui_init(window)

    try:
        print("Loading Whisper Speech-to-Text model...")
        whisper_model = load_whisper()

        print("Locating JARVIS Wake Word Model...")
        oww_model = load_wake_model()

        audio = pyaudio.PyAudio()
        mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Startup failed: {e}")
        ui_message(window, "jarvis", f"Startup failed: {e}")
        ui_status(window, "Error")
        return

    get_app_index()
    threading.Thread(target=_warm_brain, daemon=True).start()
    print("\n🟢 JARVIS is fully active. Waiting for 'Hey Jarvis'...")

    while True:
        try:
            audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            prediction = oww_model.predict(audio_data)

            if not any(score > WAKE_THRESHOLD for score in prediction.values()):
                continue

            print("\n*** WAKE WORD DETECTED! ***")
            ui_status(window, "Listening...")
            flush_mic(mic_stream)

            # --- play earcon beep immediately after wake-word detection ---
            _play_beep()

            max_idle_minutes = 20 if mem_count() else 5   # longer sessions once memory is used
            idle_chunks = 0
            retry_count = 0   # retries when audio captured but Whisper returns empty
            turn = 0
            while True:
                turn += 1
                # Silence cutoff: 0.6 s in fast/buddy mode, 0.8 s in normal mode.
                # This gives 1-2 s real-world cutoff after the user stops speaking.
                _buddy = current_personality() == "buddy"
                _sil   = 0.6 if _buddy else 0.8
                _beam  = 1   if _buddy else WHISPER_BEAM
                audio_file = record_audio(mic_stream, max_seconds=20, silence_seconds=_sil, start_timeout=8.0)
                if not audio_file:
                    # Absolutely no speech energy detected — count as idle
                    retry_count = 0
                    idle_chunks += 1
                    if idle_chunks > 6:   # ~7+ seconds of silence → sleep
                        ui_message(window, "jarvis", "Going to sleep. Say Hey Jarvis to wake me.")
                        speak("Going to sleep. Say Hey Jarvis to wake me.")
                        break
                    continue

                ui_status(window, "Transcribing...")
                segments, _ = whisper_model.transcribe(
                    audio_file,
                    language="en",                    # lock to English — prevents accent-noise drift
                    beam_size=_beam,                  # 1 in buddy mode (fast), 3 otherwise (accurate)
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=WHISPER_PROMPT,
                    no_speech_threshold=0.6,          # drop segments flagged as silence
                    log_prob_threshold=-1.0,          # drop very low-confidence segments
                    compression_ratio_threshold=2.4,  # drop repetition artifacts (Whisper loops)
                )
                parts = []
                for seg in segments:
                    if getattr(seg, "no_speech_prob", 0.0) > 0.6:
                        continue
                    if getattr(seg, "avg_logprob", 0.0) < -1.0:
                        continue
                    parts.append(seg.text)
                command = _strip_hallucination("".join(parts))
                print(f"You said: {command!r}")

                if os.path.exists(audio_file):
                    try: os.remove(audio_file)
                    except OSError: pass

                if not command:
                    # Audio was recorded but Whisper found nothing useful.
                    # Give the user up to VOICE_RETRY_MAX silent re-listens before
                    # counting it as an idle tick (avoids sleeping on a single mishear).
                    if retry_count < VOICE_RETRY_MAX:
                        retry_count += 1
                        print(f"Empty transcription — retry {retry_count}/{VOICE_RETRY_MAX}")
                        ui_status(window, "Listening...")
                        continue
                    retry_count = 0
                    idle_chunks += 1
                    if idle_chunks > 6:
                        ui_message(window, "jarvis", "Going to sleep. Say Hey Jarvis to wake me.")
                        speak("Going to sleep. Say Hey Jarvis to wake me.")
                        break
                    continue

                # Got a valid transcription — reset all counters
                retry_count = 0
                idle_chunks = 0

                norm_cmd = normalize(command)
                # Full shutdown (kills the process)
                if _is_jarvis_exit(norm_cmd):
                    speak("Shutting down. Goodbye, sir.")
                    import os as _os
                    _os._exit(0)
                # End session (goes back to wake-word standby)
                if norm_cmd in STOP_PHRASES or is_end_session(norm_cmd):
                    ui_message(window, "jarvis", "Standing by. Say Hey Jarvis when you need me.")
                    speak("Standing by. Say Hey Jarvis when you need me.")
                    break

                process_command(command, window)
                ui_status(window, "Listening for next command...")

            # Clear the wake-word model's internal buffer and stale mic audio so
            # the tail of the session can't immediately re-trigger it.
            if hasattr(oww_model, "reset"):
                oww_model.reset()
            flush_mic(mic_stream)
            ui_status(window, "Standby")

        except Exception as e:
            print(f"Backend loop error: {e}")
            ui_status(window, "Error - recovering")
            time.sleep(1)


if __name__ == "__main__":
    bootstrap_memory()                      # reload past conversations, facts and actions
    window = webview.create_window(
        "JARVIS AI Control Center",
        html=HTML_GUI,
        js_api=Api(),
        width=1100,
        height=760,
        background_color="#040302",
    )
    WINDOW = window

    backend_thread = threading.Thread(target=run_jarvis_backend, args=(window,), daemon=True)
    backend_thread.start()
    webview.start()