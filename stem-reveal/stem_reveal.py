#!/usr/bin/env python3
"""
Stem Reveal — builds a vertical "song built layer by layer" video.

  MUFFLED VOCALS 0/5 -> VOCALS 1/5 -> + MELODY 2/5 -> + BASS 3/5 -> + DRUMS 4/5 -> FULL SONG 5/5

Every change lands on a downbeat (bar start) detected in the song, the comet on
each ring completes its lap exactly when the next layer comes in, and the final
"FULL SONG" section restarts the hook on the drop with a live radial spectrum.

Usage:
  python3 stem_reveal.py song.mp3 -o out.mp4 --handle @me
  python3 stem_reveal.py song.mp3 --start 62.4          # force where the hook starts (seconds)
  python3 stem_reveal.py --stems my_stems/ -o out.mp4   # vocals/other/bass/drums .wav already split
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
from scipy import signal

SR = 44100
FPS = 30
W, H = 1080, 1920
STEMS = ("vocals", "other", "bass", "drums")

TITLES = ["MUFFLED VOCALS", "VOCALS", "+ MELODY", "+ BASS", "+ DRUMS", "FULL SONG"]
RING_LABELS = ["VOCALS", "MELODY", "BASS", "DRUMS", "DROP"]
STAGE_BARS = [2, 4, 4, 4, 4]          # length of each build stage, in bars
# which stems are audible in each build stage (stage 0 = muffled vocals)
STAGE_MIX = [
    {"vocals": 1.0},
    {"vocals": 1.0},
    {"vocals": 1.0, "other": 1.0},
    {"vocals": 1.0, "other": 1.0, "bass": 1.0},
    {"vocals": 1.0, "other": 1.0, "bass": 1.0, "drums": 1.0},
]

FONT_DIR = "/usr/share/fonts/opentype/inter"


def log(*a):
    print("[stem-reveal]", *a, flush=True)


# ----------------------------------------------------------------------------- audio io
def decode(path):
    """Any audio/video file -> float32 (n, 2) at 44.1 kHz, via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def write_wav(path, x):
    import soundfile as sf
    sf.write(path, np.clip(x, -1, 1), SR, subtype="PCM_16")


# ----------------------------------------------------------------------------- separation
def separate(mix, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    paths = {s: os.path.join(cache_dir, f"{s}.npy") for s in STEMS}
    if all(os.path.exists(p) for p in paths.values()):
        log("stems: using cache", cache_dir)
        return {s: np.load(p) for s, p in paths.items()}
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    torch.set_num_threads(os.cpu_count() or 4)
    log("stems: separating with Demucs (htdemucs) — takes a few minutes on CPU…")
    t0 = time.time()
    model = get_model("htdemucs")
    model.eval()
    wav = torch.tensor(mix.T, dtype=torch.float32)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device="cpu", split=True,
                          overlap=0.25, progress=True, num_workers=0)[0]
    out = out * std + mean
    stems = {}
    for i, name in enumerate(model.sources):
        if name in STEMS:
            stems[name] = out[i].numpy().T.astype(np.float32)
            np.save(paths[name], stems[name])
    log(f"stems: done in {time.time() - t0:.0f}s")
    return stems


def load_stem_dir(d):
    stems = {}
    for s in STEMS:
        cands = [f for f in os.listdir(d) if os.path.splitext(f)[0].lower() == s]
        if not cands:
            sys.exit(f"missing stem '{s}' in {d} (expected {s}.wav / {s}.mp3 …)")
        stems[s] = decode(os.path.join(d, cands[0]))
    n = min(len(v) for v in stems.values())
    return {k: v[:n] for k, v in stems.items()}


# ----------------------------------------------------------------------------- analysis
def frame_rms(x, hop):
    m = x.mean(1) if x.ndim == 2 else x
    n = len(m) // hop
    return np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)


def analyze(mix, stems):
    import librosa
    sr2 = 22050
    mono = librosa.resample(mix.mean(1), orig_sr=SR, target_sr=sr2)
    hop = 512
    onset = librosa.onset.onset_strength(y=mono, sr=sr2, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr2, hop_length=hop,
                                                 tightness=120, trim=False)
    beats = librosa.frames_to_time(beat_frames, sr=sr2, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 32:
        sys.exit("could not find a steady beat in this song")

    # --- downbeat phase: kick energy + harmonic change both peak on bar starts
    drums = librosa.resample(stems["drums"].mean(1), orig_sr=SR, target_sr=sr2)
    sos = signal.butter(4, 150, "lowpass", fs=sr2, output="sos")
    kick_env = np.abs(signal.sosfilt(sos, drums))
    harm = librosa.resample((stems["other"] + stems["bass"]).mean(1), orig_sr=SR, target_sr=sr2)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=sr2, hop_length=hop)
    bf = np.clip(librosa.time_to_frames(beats, sr=sr2, hop_length=hop), 0, chroma.shape[1] - 1)
    cb = chroma[:, bf]
    chg = np.r_[0, np.linalg.norm(np.diff(cb, axis=1), axis=0)]
    kick = np.array([kick_env[int(b * sr2): int(b * sr2) + int(0.08 * sr2)].mean() for b in beats])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    kz, cz = z(kick), z(chg)
    phase_score = [kz[p::4].mean() + 1.5 * cz[p::4].mean() for p in range(4)]
    phase = int(np.argmax(phase_score))
    bars = beats[phase::4]
    beat_dur = float(np.median(np.diff(beats)))
    log(f"tempo ≈ {60 / beat_dur:.1f} BPM, {len(bars)} bars, downbeat phase {phase}")
    return beats, bars, beat_dur


def find_hook(mix, stems, bars, need_bars):
    """Bar index where the strongest vocal section (chorus / drop) starts."""
    def per_bar(x):
        m = x.mean(1)
        out = []
        for a, b in zip(bars[:-1], bars[1:]):
            seg = m[int(a * SR): int(b * SR)]
            out.append(np.sqrt((seg ** 2).mean() + 1e-12))
        return np.array(out)

    tot, voc = per_bar(mix), per_bar(stems["vocals"])
    low = per_bar(stems["drums"]) + per_bar(stems["bass"])
    n = len(tot)
    tot_n, voc_n, low_n = tot / tot.max(), voc / voc.max(), low / (low.max() + 1e-9)
    best, best_j = -1e9, None
    for j in range(4, n - 8):
        if j + need_bars > n:
            break
        nxt = slice(j, j + 8)
        prev = slice(max(0, j - 4), j)
        e_next, e_prev = tot_n[nxt].mean(), tot_n[prev].mean()
        jump = np.clip(20 * np.log10((e_next + 1e-6) / (e_prev + 1e-6)), 0, 8) / 8
        score = (1.6 * e_next + 1.0 * voc_n[nxt].mean() + 0.6 * low_n[nxt].mean() + 0.7 * jump
                 - 0.25 * j / n + (0.15 if j % 4 == 0 else 0))
        if voc_n[nxt].min() < 0.08:          # the hook needs vocals all the way
            score -= 0.6
        if score > best:
            best, best_j = score, j
    if best_j is None:
        best_j = max(0, min(n - need_bars, n // 3))
    # how long the loud vocal section lasts -> loop length for the build-up
    run = 0
    ref = tot_n[best_j: best_j + 4].mean()
    while best_j + run < n and tot_n[best_j + run] > 0.55 * ref and voc_n[best_j + run] > 0.06:
        run += 1
    return best_j, run


# ----------------------------------------------------------------------------- arrangement
def lowpass(x, f):
    sos = signal.butter(2, f, "lowpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


def hp_sweep(x, f0, f1):
    """Rising high-pass filter over the whole block (classic pre-drop build)."""
    out = np.empty_like(x)
    blk = 512
    n = len(x)
    zi = None
    for i in range(0, n, blk):
        f = f0 * (f1 / f0) ** (i / max(1, n - 1))
        sos = signal.butter(2, f, "highpass", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, 2))
        out[i: i + blk], zi = signal.sosfilt(sos, x[i: i + blk], axis=0, zi=zi)
    return out


def arrange(mix, stems, beats, bars, hook, run, min_duration):
    bar_dur = float(np.median(np.diff(bars)))
    stage_bars = list(STAGE_BARS)
    if bar_dur < 1.6:                                  # very fast song: longer stages
        stage_bars = [b * 2 for b in stage_bars]
    pre_bars = sum(stage_bars)
    loop = None if run >= pre_bars else (16 if run >= 16 else 8)

    def bar_span(i):
        a = bars[i] if i < len(bars) else bars[-1] + (i - len(bars) + 1) * bar_dur
        b = bars[i + 1] if i + 1 < len(bars) else a + bar_dur
        return a, b

    # build-up bars (looping inside the hook if it is short)
    segments = []        # (src_a, src_b, stage)
    b = 0
    for s, nb in enumerate(stage_bars):
        for _ in range(nb):
            src = hook + (b % loop if loop else b)
            segments.append((*bar_span(src), s))
            b += 1
    t_pre = sum(e - a for a, e, _ in segments)
    # full song: restart on the hook, enough bars to pass min_duration
    full = 0
    while True:
        full += 4
        dur = sum(bar_span(hook + k)[1] - bar_span(hook + k)[0] for k in range(full))
        if t_pre + dur >= min_duration or hook + full >= len(bars) - 1:
            break
    for k in range(full):
        segments.append((*bar_span(hook + k), 5))
    total = t_pre + sum(bar_span(hook + k)[1] - bar_span(hook + k)[0] for k in range(full))
    if total < min_duration:
        log(f"warning: song too short after the hook, video is {total:.1f}s")

    # ---- render audio, segment by segment, with 10 ms crossfades
    X = int(0.010 * SR)
    n_out = int(round(total * SR)) + 2 * X
    out = np.zeros((n_out, 2), np.float32)
    muffled = lowpass(stems["vocals"], 650)
    sources = dict(stems, muffled=muffled)
    n_src = len(mix)

    def take(arr, a, b):
        a0, b0 = max(0, a), min(n_src, b)
        seg = np.zeros((b - a, 2), np.float32)
        if b0 > a0:
            seg[a0 - a: b0 - a] = arr[a0:b0]
        return seg

    def stage_audio(s, a, b):
        if s == 5:
            return take(mix, a, b)
        acc = np.zeros((b - a, 2), np.float32)
        for name, g in STAGE_MIX[s].items():
            src = sources["muffled" if (s == 0 and name == "vocals") else name]
            acc += g * take(src, a, b)
        return acc

    # loudness compensation so quiet early stages are not much softer than the drop
    full_rms = np.sqrt((mix[int(bars[hook] * SR): int(bar_span(hook + 8)[1] * SR)] ** 2).mean() + 1e-9)
    stage_gain = {}
    for s in range(5):
        segs = [(a, e) for a, e, st in segments if st == s]
        acc = np.concatenate([stage_audio(s, int(a * SR), int(e * SR)) for a, e in segs])
        r = np.sqrt((acc ** 2).mean() + 1e-9)
        cap = 15 if s == 0 else 7          # muffled vocals lose most of their energy
        stage_gain[s] = float(np.clip(full_rms * 10 ** ((-4 if s == 0 else -2.5) / 20) / r, 1.0, 10 ** (cap / 20)))
    stage_gain[5] = 1.0

    starts, pos = [], 0
    win_in = np.linspace(0, 1, 2 * X, dtype=np.float32)[:, None]
    stage_times = [None] * 6
    src_map = []                                        # (out_t0, out_t1, src_t0) for beat mapping
    for i, (a, e, s) in enumerate(segments):
        ia, ie = int(round(a * SR)), int(round(e * SR))
        seg = stage_audio(s, ia - X, ie + X) * stage_gain[s]
        seg[: 2 * X] *= win_in
        seg[-2 * X:] *= win_in[::-1]
        L = min(len(seg), n_out - pos)
        out[pos: pos + L] += seg[:L]
        if stage_times[s] is None:
            stage_times[s] = pos / SR
        src_map.append((pos / SR, (pos + ie - ia) / SR, a))
        pos += ie - ia
    out = out[X: X + pos]

    # pre-drop: rising high-pass over the last 2 bars of "+ DRUMS", short gap before the drop
    d0, d1 = int(stage_times[4] * SR), int(stage_times[5] * SR)
    sweep_start = d0 + int((d1 - d0) * (1 - 2 / stage_bars[4]))
    out[sweep_start:d1] = hp_sweep(out[sweep_start:d1], 40, 900)
    gap = int(0.5 * float(np.median(np.diff(beats))) * SR)
    fade = int(0.02 * SR)
    g = np.ones(d1 - (d1 - gap), np.float32)
    g[:fade] = np.linspace(1, 0, fade)
    g[fade:] = 0
    out[d1 - gap: d1] *= g[:, None]
    # outro fade
    nf = int(1.6 * SR)
    out[-nf:] *= np.linspace(1, 0, nf, dtype=np.float32)[:, None] ** 1.5
    # gentle limiter
    peak = np.percentile(np.abs(out), 99.995)
    out = out * (0.92 / max(peak, 1e-6))
    out = np.where(np.abs(out) > 0.9, np.sign(out) * (0.9 + 0.1 * np.tanh((np.abs(out) - 0.9) / 0.1)), out)

    # beats & downbeats on the output timeline
    out_beats, out_down = [], []
    bar_set = set(np.round(bars, 4))
    for t0, t1, sa in src_map:
        for bt in beats:
            if sa - 1e-3 <= bt < sa + (t1 - t0) - 1e-3:
                ot = t0 + (bt - sa)
                out_beats.append(ot)
                if round(bt, 4) in bar_set:
                    out_down.append(ot)
    plan = dict(
        bpm=round(60 / float(np.median(np.diff(beats))), 1),
        hook_start_s=round(float(bars[hook]), 2),
        loop_bars=loop, stage_bars=stage_bars, full_bars=full,
        stage_starts_s=[round(t, 3) for t in stage_times],
        duration_s=round(len(out) / SR, 2),
        stage_gain_db=[round(20 * math.log10(stage_gain[s]), 1) for s in range(6)],
    )
    return out.astype(np.float32), stage_times, np.array(out_beats), np.array(out_down), plan


# ----------------------------------------------------------------------------- per-frame audio features
def audio_features(audio, n_frames, stage5_t):
    hop = SR // FPS
    mono = audio.mean(1)
    # kick pulse: positive flux of the low band
    low = signal.sosfilt(signal.butter(2, 160, "lowpass", fs=SR, output="sos"), mono)
    env = np.array([np.sqrt((low[i * hop:(i + 1) * hop] ** 2).mean() + 1e-12) for i in range(n_frames)])
    full = np.array([np.sqrt((mono[i * hop:(i + 1) * hop] ** 2).mean() + 1e-12) for i in range(n_frames)])
    flux = np.maximum(0, np.diff(np.r_[env[0], env]))
    flux /= np.percentile(flux, 97) + 1e-9
    pulse = np.zeros(n_frames)
    for i in range(n_frames):
        pulse[i] = max(min(flux[i], 1.3), (pulse[i - 1] if i else 0) * 0.80)
    level = full / (np.percentile(full, 95) + 1e-9)

    # radial spectrum bands for the FULL SONG section
    nb = 72
    nfft = 4096
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    edges = np.geomspace(35, 14000, nb + 1)
    idx = [np.where((freqs >= edges[k]) & (freqs < edges[k + 1]))[0] for k in range(nb)]
    idx = [ix if len(ix) else np.array([np.argmin(np.abs(freqs - edges[k]))]) for k, ix in enumerate(idx)]
    win = np.hanning(nfft)
    spec = np.zeros((n_frames, nb), np.float32)
    f0 = int(stage5_t * FPS) - 2
    for i in range(max(0, f0), n_frames):
        c = i * hop + hop // 2
        seg = mono[max(0, c - nfft // 2): c + nfft // 2]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win))
        spec[i] = [20 * np.log10(mag[ix].mean() + 1e-9) for ix in idx]
    sec = spec[max(0, f0):]
    if len(sec):
        top = np.percentile(sec, 97, axis=0)
        spec = np.clip((spec - (top - 32)) / 32, 0, 1) ** 1.6
        sm = np.zeros_like(spec)
        for i in range(1, n_frames):
            a = np.where(spec[i] > sm[i - 1], 0.65, 0.18)
            sm[i] = sm[i - 1] + a * (spec[i] - sm[i - 1])
        spec = sm
    spec[: max(0, f0)] = 0
    return pulse.astype(np.float32), level.astype(np.float32), spec


# ----------------------------------------------------------------------------- rendering
G = {}   # per-process render state


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def init_render(state):
    import skia
    G.update(state)
    G["skia"] = skia
    tf = lambda f: skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, f))
    G["f_title"] = skia.Font(tf("InterDisplay-ExtraBold.otf"), 92)
    G["f_count"] = skia.Font(tf("Inter-SemiBold.otf"), 34)
    G["f_small"] = skia.Font(tf("Inter-SemiBold.otf"), 30)
    G["f_tag"] = skia.Font(tf("Inter-Bold.otf"), 22)
    G["f_cta"] = skia.Font(tf("InterDisplay-ExtraBold.otf"), 40)
    G["f_cta2"] = skia.Font(tf("Inter-Medium.otf"), 28)
    for f in ("f_title", "f_count", "f_small", "f_tag", "f_cta", "f_cta2"):
        G[f].setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    rng = np.random.default_rng(7)
    G["bubbles"] = np.c_[rng.uniform(0, W, 75), rng.uniform(0, H, 75), rng.uniform(14, 42, 75),
                         rng.uniform(3, 13, 75), rng.uniform(0.07, 0.26, 75), rng.uniform(0, 6.28, 75)]
    bursts = []
    for s in range(1, 6):
        r = np.random.default_rng(100 + s)
        n = 170
        bursts.append(np.c_[r.uniform(0, 2 * np.pi, n), r.uniform(-60, 160, n), r.uniform(-90, 90, n),
                            r.uniform(0.6, 1.5, n), r.uniform(1.4, 3.2, n)])
    G["bursts"] = bursts
    G["surface"] = skia.Surface(W, H)


def text_c(canvas, s, font, x, y, paint):
    w = font.measureText(s)
    canvas.drawString(s, x - w / 2, y, font, paint)


def rgba(r, g, b, a):
    skia = G["skia"]
    return skia.Color4f(r / 255, g / 255, b / 255, float(min(max(a, 0), 1)))


def render_frame(i):
    skia = G["skia"]
    t = i / FPS
    st = G["stage_times"]
    stage = max(k for k in range(6) if st[k] <= t + 1e-6)
    ts = st[stage]
    te = st[stage + 1] if stage < 5 else G["duration"]
    p = (t - ts) / max(1e-6, te - ts)
    pulse = float(G["pulse"][min(i, len(G["pulse"]) - 1)])
    level = float(G["level"][min(i, len(G["level"]) - 1)])
    accent = G["accent"]
    cx, cy = W / 2, 1010

    surface = G["surface"]
    c = surface.getCanvas()
    # background
    bg = skia.Paint(Shader=skia.GradientShader.MakeLinear(
        [skia.Point(0, 0), skia.Point(0, H)], [skia.Color(13, 18, 26), skia.Color(5, 7, 11)]))
    c.drawRect(skia.Rect(0, 0, W, H), bg)
    vign = skia.Paint(Shader=skia.GradientShader.MakeRadial(
        skia.Point(cx, cy), 900, [skia.Color(22, 30, 44, int(90 + 60 * pulse)), skia.Color(0, 0, 0, 0)]))
    c.drawRect(skia.Rect(0, 0, W, H), vign)

    # bubbles
    bp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.8)
    for x0, y0, sp, r, a, ph in G["bubbles"]:
        y = (y0 - sp * t) % (H + 60) - 30
        x = x0 + 14 * math.sin(t * 0.6 + ph)
        bp.setColor4f(rgba(190, 210, 235, a * (0.8 + 0.5 * pulse)))
        c.drawCircle(x, y, r * (1 + 0.12 * pulse), bp)

    glow = lambda sigma: skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma)

    if stage < 5:
        R0, D = 250, 50
        z = (stage - 1) + ease_out((t - ts) / 0.45) if stage > 0 else 0.0
        # center shade
        c.drawCircle(cx, cy, R0 * 0.62, skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeRadial(
            skia.Point(cx, cy), R0 * 0.62, [skia.Color(3, 4, 7, 235), skia.Color(3, 4, 7, 0)])))
        for k in range(4, -1, -1):
            r = R0 + (k - z) * D
            if k < stage:                                   # ring just completed: collapse
                fade = 1 - min(1, (z - k) * 1.6)
                if fade <= 0:
                    continue
                pa = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4)
                pa.setColor4f(rgba(255, 255, 255, 0.9 * fade))
                c.drawCircle(cx, cy, max(10, r), pa)
                continue
            rect = skia.Rect(cx - r, cy - r, cx + r, cy + r)
            if k == stage:
                base = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3)
                base.setColor4f(rgba(210, 222, 240, 0.20 + 0.10 * pulse))
                c.drawCircle(cx, cy, r, base)
                sweep = 360 * min(1, p)
                for sig, wdt, al in ((18, 14, 0.35 + 0.35 * pulse), (5, 7, 0.55), (0, 4.5, 1.0)):
                    pa = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=wdt,
                                    StrokeCap=skia.Paint.kRound_Cap)
                    pa.setColor4f(rgba(*accent, al) if sig == 18 else rgba(255, 255, 255, al))
                    if sig:
                        pa.setMaskFilter(glow(sig))
                    c.drawArc(rect, -90, sweep, False, pa)
                # comet: trail + head
                ang = math.radians(-90 + sweep)
                for j in range(26):
                    aa = ang - math.radians(j * 1.6)
                    pa = skia.Paint(AntiAlias=True)
                    pa.setColor4f(rgba(255, 255, 255, 0.9 * (1 - j / 26) ** 2))
                    c.drawCircle(cx + r * math.cos(aa), cy + r * math.sin(aa), 7 * (1 - j / 30), pa)
                hp = skia.Paint(AntiAlias=True, MaskFilter=glow(14 + 10 * pulse))
                hp.setColor4f(rgba(*accent, 0.9))
                c.drawCircle(cx + r * math.cos(ang), cy + r * math.sin(ang), 16 + 8 * pulse, hp)
                hp2 = skia.Paint(AntiAlias=True)
                hp2.setColor4f(rgba(255, 255, 255, 1))
                c.drawCircle(cx + r * math.cos(ang), cy + r * math.sin(ang), 8, hp2)
            else:                                           # upcoming rings: rotating broken arcs
                al = max(0.0, 0.30 - 0.05 * (k - z - 1))
                pa = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.6,
                                StrokeCap=skia.Paint.kRound_Cap)
                pa.setColor4f(rgba(200, 212, 232, al))
                rot = t * (14 + 6 * k) * (1 if k % 2 else -1) + 40 * k
                gap = 46 + 22 * math.sin(t * 0.7 + k)
                c.drawArc(rect, rot, 360 - gap, False, pa)
            # ring tag
            lab = RING_LABELS[k]
            tw = G["f_tag"].measureText(lab)
            tag = skia.RRect.MakeRectXY(skia.Rect(cx - tw / 2 - 12, cy - r - 17, cx + tw / 2 + 12, cy - r + 17), 6, 6)
            act = (k == stage)
            ta = 1.0 if act else max(0.0, 0.55 - 0.1 * (k - z - 1))
            fp = skia.Paint(AntiAlias=True)
            fp.setColor4f(rgba(235, 240, 250, ta) if act else rgba(30, 38, 52, ta))
            c.drawRRect(tag, fp)
            if not act:
                sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.5)
                sp.setColor4f(rgba(120, 135, 160, ta * 0.8))
                c.drawRRect(tag, sp)
            tp = skia.Paint(AntiAlias=True)
            tp.setColor4f(rgba(12, 16, 24, 1) if act else rgba(170, 182, 200, ta))
            c.drawString(lab, cx - tw / 2, cy - r + 8, G["f_tag"], tp)

        # burst of particles when a layer arrives
        if stage > 0 and t - ts < 1.6:
            dt = t - ts
            rr = R0 + (stage - 1 - z) * D
            pp = skia.Paint(AntiAlias=True)
            for a0, vr, vt, life, sz in G["bursts"][stage - 1]:
                if dt > life:
                    continue
                k_ = dt / life
                rad = rr + vr * dt
                a = a0 + vt * dt / max(rad, 40) * 0.6
                pp.setColor4f(rgba(255, 255, 255, 0.85 * (1 - k_) ** 1.5))
                c.drawCircle(cx + rad * math.cos(a), cy + rad * math.sin(a), sz, pp)
    else:
        # FULL SONG: radial spectrum
        dt = t - ts
        spec = G["spec"][min(i, len(G["spec"]) - 1)]
        nb = len(spec)
        appear = ease_out(dt / 0.5)
        Rc = (150 + 18 * pulse) * (0.6 + 0.4 * appear)
        sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.2,
                        StrokeCap=skia.Paint.kRound_Cap)
        gp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=7,
                        StrokeCap=skia.Paint.kRound_Cap, MaskFilter=glow(9))
        gp.setColor4f(rgba(*accent, 0.35 * appear))
        sp.setColor4f(rgba(225, 232, 245, 0.75 * appear))
        path_g, path_s = skia.Path(), skia.Path()
        for side in (-1, 1):
            for k in range(nb):
                ang = math.radians(-90 + side * (k + 0.5) / nb * 180)
                ln = (10 + 300 * float(spec[k])) * appear
                x0, y0 = cx + (Rc + 8) * math.cos(ang), cy + (Rc + 8) * math.sin(ang)
                x1, y1 = cx + (Rc + 8 + ln) * math.cos(ang), cy + (Rc + 8 + ln) * math.sin(ang)
                for pth in (path_g, path_s):
                    pth.moveTo(x0, y0)
                    pth.lineTo(x1, y1)
        c.drawPath(path_g, gp)
        c.drawPath(path_s, sp)
        cp = skia.Paint(AntiAlias=True)
        cp.setColor4f(rgba(4, 5, 8, 1))
        c.drawCircle(cx, cy, Rc, cp)
        rp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=6)
        rp.setColor4f(rgba(70, 80, 98, 0.9))
        c.drawCircle(cx, cy, Rc, rp)
        rp2 = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3, MaskFilter=glow(10))
        rp2.setColor4f(rgba(*accent, 0.25 + 0.5 * pulse))
        c.drawCircle(cx, cy, Rc + 4, rp2)
        # drop flash
        if dt < 0.45:
            fl = skia.Paint()
            fl.setColor4f(rgba(255, 255, 255, 0.45 * (1 - dt / 0.45) ** 2))
            c.drawRect(skia.Rect(0, 0, W, H), fl)
        # call to action
        ca = ease_out((dt - 3.0) / 0.6)
        if ca > 0:
            p1 = skia.Paint(AntiAlias=True)
            p1.setColor4f(rgba(*accent, ca))
            text_c(c, G["cta"][0], G["f_cta"], cx, 1500 + 20 * (1 - ca), p1)
            p2 = skia.Paint(AntiAlias=True)
            p2.setColor4f(rgba(150, 160, 178, ca))
            text_c(c, G["cta"][1], G["f_cta2"], cx, 1548 + 20 * (1 - ca), p2)

    # header: handle, title (pops on each change), counter
    hp_ = skia.Paint(AntiAlias=True)
    hp_.setColor4f(rgba(150, 160, 178, 0.75))
    if G["handle"]:
        text_c(c, G["handle"], G["f_small"], cx, 228, hp_)
    pop = 1 + 0.22 * math.exp(-(t - ts) * 9) * (1 if stage else 0)
    ti = skia.Paint(AntiAlias=True)
    ti.setColor4f(rgba(255, 255, 255, min(1, 0.35 + (t - ts) * 4) if stage else 1))
    c.save()
    c.translate(cx, 322)
    c.scale(pop, pop)
    text_c(c, TITLES[stage], G["f_title"], 0, 0, ti)
    c.restore()
    cp_ = skia.Paint(AntiAlias=True)
    cp_.setColor4f(rgba(165, 175, 192, 0.9))
    text_c(c, f"{stage} / 5", G["f_count"], cx, 380, cp_)
    if stage == 5 and G["credit"]:
        cr = skia.Paint(AntiAlias=True)
        cr.setColor4f(rgba(150, 160, 178, ease_out((t - ts - 0.6) / 0.6)))
        text_c(c, G["credit"], G["f_small"], cx, 430, cr)

    # fade in / out
    fade = min(1.0, t / 0.25, (G["duration"] - t) / 1.2)
    if fade < 1:
        fp = skia.Paint()
        fp.setColor4f(rgba(0, 0, 0, 1 - max(0.0, fade)))
        c.drawRect(skia.Rect(0, 0, W, H), fp)
    return surface.makeImageSnapshot().toarray().tobytes()


def render_video(audio, stage_times, args, plan, out_path, workdir):
    duration = len(audio) / SR
    n_frames = int(duration * FPS)
    pulse, level, spec = audio_features(audio, n_frames, stage_times[5])
    wav = os.path.join(workdir, "mix.wav")
    write_wav(wav, audio)
    accent = tuple(int(args.accent.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
    state = dict(stage_times=stage_times, duration=duration, pulse=pulse, level=level, spec=spec,
                 accent=accent, handle=args.handle, credit=args.credit,
                 cta=(args.cta, args.cta2))
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
         out_path], stdin=subprocess.PIPE)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(max(1, (os.cpu_count() or 2) - 1), initializer=init_render, initargs=(state,)) as pool:
        for k, buf in enumerate(pool.imap(render_frame, range(n_frames), chunksize=6)):
            ff.stdin.write(buf)
            if k % 150 == 0:
                log(f"render {k}/{n_frames} frames ({time.time() - t0:.0f}s)")
    ff.stdin.close()
    ff.wait()
    log(f"video written: {out_path} ({duration:.1f}s, {time.time() - t0:.0f}s render)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song", nargs="?", help="song file (mp3, wav, m4a, even a video)")
    ap.add_argument("--stems", help="folder with vocals/other/bass/drums files (skips separation)")
    ap.add_argument("-o", "--out", default="stem_reveal.mp4")
    ap.add_argument("--start", type=float, help="force the hook start time in seconds")
    ap.add_argument("--min-duration", type=float, default=62.0, help="minimum video length (s)")
    ap.add_argument("--handle", default="@yourhandle")
    ap.add_argument("--credit", default="", help='e.g. "Song Title — Artist" (shown on FULL SONG)')
    ap.add_argument("--accent", default="#9FB4FF")
    ap.add_argument("--cta", default="FOLLOW FOR THE NEXT DROP")
    ap.add_argument("--cta2", default="new songs every day")
    ap.add_argument("--workdir", default=".stem_reveal_cache")
    args = ap.parse_args()
    if not args.song and not args.stems:
        ap.error("give a song file or --stems")

    if args.stems:
        stems = load_stem_dir(args.stems)
        mix = sum(stems.values())
        key = hashlib.md5(os.path.abspath(args.stems).encode()).hexdigest()[:10]
    else:
        mix = decode(args.song)
        key = hashlib.md5(open(args.song, "rb").read(1 << 20) + str(len(mix)).encode()).hexdigest()[:10]
    work = os.path.join(args.workdir, key)
    os.makedirs(work, exist_ok=True)
    if not args.stems:
        stems = separate(mix, work)
        n = min(len(mix), *(len(v) for v in stems.values()))
        mix, stems = mix[:n], {k: v[:n] for k, v in stems.items()}

    beats, bars, _ = analyze(mix, stems)
    need = sum(STAGE_BARS) // 2
    if args.start is not None:
        hook = int(np.argmin(np.abs(bars - args.start)))
        run = 8
    else:
        hook, run = find_hook(mix, stems, bars, need)
    log(f"hook starts at {bars[hook]:.2f}s (bar {hook}), loud vocal section ≈ {run} bars")

    audio, stage_times, out_beats, out_down, plan = arrange(mix, stems, beats, bars, hook, run, args.min_duration)
    plan["out"] = args.out
    log("plan:", json.dumps(plan))
    with open(os.path.splitext(args.out)[0] + ".plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    render_video(audio, stage_times, args, plan, args.out, work)


if __name__ == "__main__":
    main()
