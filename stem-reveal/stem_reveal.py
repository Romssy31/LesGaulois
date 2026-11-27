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

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
if not os.path.exists(os.path.join(FONT_DIR, "Inter-Bold.otf")):
    FONT_DIR = "/usr/share/fonts/opentype/inter"


def ffmpeg_exe():
    """System ffmpeg if installed, else the one shipped with the imageio-ffmpeg package."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def log(*a):
    print("[stem-reveal]", *a, flush=True)


# ----------------------------------------------------------------------------- audio io
def decode(path):
    """Any audio/video file -> float32 (n, 2) at 44.1 kHz, via ffmpeg."""
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def write_wav(path, x):
    import soundfile as sf
    sf.write(path, np.clip(x, -1, 1), SR, subtype="PCM_16")


# ----------------------------------------------------------------------------- separation
def separate(mix, cache_dir, progress=None):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import demucs.apply as dapply
    orig_tqdm = dapply.tqdm.tqdm

    def tracked(it, *a, **k):                     # report Demucs progress to the caller
        items = list(it)
        for n_, x in enumerate(items):
            if progress:
                progress(n_ / max(1, len(items)))
            yield x
    dapply.tqdm.tqdm = tracked
    try:
        with torch.no_grad():
            out = apply_model(model, ((wav - mean) / std)[None], device=device, split=True,
                              overlap=0.25, progress=True, num_workers=0)[0]
    finally:
        dapply.tqdm.tqdm = orig_tqdm
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


def bar_features(mix, stems, bars):
    def per_bar(x):
        m = x.mean(1)
        out = []
        for a, b in zip(bars[:-1], bars[1:]):
            seg = m[int(a * SR): int(b * SR)]
            out.append(np.sqrt((seg ** 2).mean() + 1e-12))
        return np.array(out)

    tot, voc = per_bar(mix), per_bar(stems["vocals"])
    low = per_bar(stems["drums"]) + per_bar(stems["bass"])
    return dict(tot=tot / tot.max(), voc=voc / voc.max(), low=low / (low.max() + 1e-9))


def section_run(feats, j):
    """How many bars the loud vocal section starting at bar j lasts."""
    tot_n, voc_n = feats["tot"], feats["voc"]
    n, run = len(tot_n), 0
    ref = tot_n[j: j + 4].mean()
    while j + run < n and tot_n[j + run] > 0.55 * ref and voc_n[j + run] > 0.06:
        run += 1
    return run


def find_hook(feats, need_bars):
    """Bar index where the strongest vocal section (chorus / drop) starts."""
    tot_n, voc_n, low_n = feats["tot"], feats["voc"], feats["low"]
    n = len(tot_n)
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
    return best_j


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


def arrange(mix, stems, beats, bars, hook, run, min_duration, max_duration=None, stage_bars=None, full_bars=None):
    """stage_bars: bars for each of the 5 build stages (None = automatic).
    full_bars: bars of the FULL SONG part (None = enough to reach min_duration).
    max_duration: the FULL SONG part is shortened (bar by bar) to stay under it."""
    bar_dur = float(np.median(np.diff(bars)))
    if stage_bars:
        stage_bars = [max(1, int(b)) for b in stage_bars][:5]
        if len(stage_bars) != 5:
            raise ValueError("il faut 5 durées d'étapes (en mesures)")
    else:
        stage_bars = list(STAGE_BARS)
        if bar_dur < 1.6:                              # very fast song: longer stages
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

    def full_dur(n):
        return sum(bar_span(hook + k)[1] - bar_span(hook + k)[0] for k in range(n))

    # full song: restarts on the hook
    avail = max(1, len(bars) - 1 - hook)               # bars of real music after the hook
    if full_bars:
        full = int(full_bars)
    else:                                              # enough bars (by 4) to pass min_duration
        full = 0
        while True:
            full += 4
            if t_pre + full_dur(full) >= min_duration or full >= avail:
                break
    full = max(1, min(full, avail))
    if max_duration:
        if t_pre + full_dur(1) > max_duration + 1e-6:
            raise ValueError(f"les 5 étapes de montée durent déjà {t_pre:.1f} s : c'est plus que la durée max "
                             f"({max_duration:.0f} s). Raccourcis des étapes ou augmente la durée max.")
        while full > 1 and t_pre + full_dur(full) > max_duration + 1e-6:
            full -= 1
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
    sweep_start = d0 + int((d1 - d0) * (1 - min(2, stage_bars[4]) / stage_bars[4]))
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
        loop_bars=loop, stage_bars=stage_bars, full_bars=full, bar_s=round(bar_dur, 3),
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


# ----------------------------------------------------------------------------- ball & rings
RING_R = [125, 200, 275, 350, 425]
BALL_R = 13
CX, CY = W / 2, 1010


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan_ball(stage_times, beats):
    """Ball path: one bounce on every beat, escape through the ring's slit on the
    exact downbeat where the next layer comes in. Each ring's slit rotates so that
    it reaches the escape point at that instant and is never under an earlier bounce."""
    beats = np.sort(np.asarray(beats, float))
    keys = [(0.0, 0.0, 0.0)]
    hits, gaps, escapes = [], [], []
    pos, direction = np.zeros(2), None
    for k in range(5):
        T0, T1 = stage_times[k], stage_times[k + 1]
        bt = [float(b) for b in beats if T0 + 0.05 < b < T1 - 0.05] + [T1]
        re = RING_R[k] - BALL_R - 3                       # ball centre radius at contact
        if k == 0:
            delta, sgn, psi0 = math.radians(137.5), 1, math.radians(-55)
        else:
            bd = pos @ direction
            sd = -bd + math.sqrt(max(0.0, bd * bd - (pos @ pos - re * re)))
            q = pos + sd * direction                      # keeps flying straight out of the slit
            psi0 = math.atan2(q[1], q[0])
            cross = pos[0] * direction[1] - pos[1] * direction[0]
            sgn = 1 if cross >= 0 else -1
            # the ring it left is gone: plain mirror reflection keeps the same line
            # offset from the centre; clamped so it never just goes back and forth
            delta = 2 * math.acos(min(0.999, abs(cross) / re))
            delta = min(max(delta, math.radians(75)), math.radians(150))
        angs = [psi0 + sgn * j * delta for j in range(len(bt))]
        pts = [np.array([re * math.cos(x), re * math.sin(x)]) for x in angs]
        for tb, pt, an in zip(bt[:-1], pts[:-1], angs[:-1]):
            keys.append((tb, pt[0], pt[1]))
            hits.append((tb, pt[0], pt[1], k, an))
        prev_t, prev = (bt[-2], pts[-2]) if len(pts) > 1 else (T0, pos)
        # the ball's centre is right in the slit (ring radius) at the exact layer change
        dd = (pts[-1] - prev) / (np.linalg.norm(pts[-1] - prev) + 1e-9)
        bd = prev @ dd
        e = prev + (-bd + math.sqrt(max(0.0, bd * bd - (prev @ prev - RING_R[k] ** 2)))) * dd
        angs[-1] = math.atan2(e[1], e[0])
        keys.append((T1, e[0], e[1]))
        direction = (e - prev) / (np.linalg.norm(e - prev) + 1e-9)
        speed = np.linalg.norm(e - prev) / max(1e-3, T1 - prev_t)
        pos = e
        escapes.append((T1, e[0], e[1], math.atan2(direction[1], direction[0]), k))
        # slit: centre angle(t) = phi + omega * (t - T1)
        half = max(math.radians(7.5), (BALL_R + 17) / RING_R[k])
        margin = (BALL_R + 12) / RING_R[k]
        phi, best = angs[-1], None
        for om_deg in (40, -40, 55, -55, 30, -30, 75, -75, 95, -95, 22, -22, 120, -120, 150, -150):
            om = math.radians(om_deg)
            dmin = min([abs(wrap(an - (phi + om * (tb - T1)))) for an, tb in zip(angs[:-1], bt[:-1])] or [9.0])
            if best is None or dmin > best[0]:
                best = (dmin, om)
            if dmin > half + margin:
                best = (dmin, om)
                break
        gaps.append((phi, best[1], T1, half))
    end = pos + direction * max(speed, 500) * 2.0
    keys.append((stage_times[5] + 2.0, end[0], end[1]))
    k_ = np.array(keys)
    return dict(kt=k_[:, 0], kx=k_[:, 1], ky=k_[:, 2], hits=hits, gaps=gaps, escapes=escapes)


def ball_pos(t):
    B = G["ball"]
    return CX + float(np.interp(t, B["kt"], B["kx"])), CY + float(np.interp(t, B["kt"], B["ky"]))


def slit_angle(k, t):
    phi, om, T1, half = G["ball"]["gaps"][k]
    return phi + om * (t - T1), half


def draw_rings(c, t, stage, fade, pulse):
    skia = G["skia"]
    accent = G["accent"]
    for k in range(5):
        r = RING_R[k]
        g, half = slit_angle(k, t)
        rect = skia.Rect(CX - r, CY - r, CX + r, CY + r)
        start, sweep = math.degrees(g + half), 360 - math.degrees(2 * half)
        if k == stage:
            layers = ((16, 12, 0.30 + 0.35 * pulse, accent), (0, 5, 1.0, (255, 255, 255)))
        elif k > stage:
            layers = ((0, 3, 0.42, (205, 215, 235)),)
        else:                                               # escaped: bursts and disappears
            dt = t - G["ball"]["gaps"][k][2]
            if dt >= 0.45:
                continue
            q = dt / 0.45
            r = r * (1 + 0.10 * ease_out(q))
            rect = skia.Rect(CX - r, CY - r, CX + r, CY + r)
            layers = ((10, 8, 0.6 * (1 - q), accent), (0, 4 * (1 - q) + 1, (1 - q) ** 1.5, (255, 255, 255)))
            dp = skia.Paint(AntiAlias=True)
            for j in range(96):
                a = g + j * 2 * math.pi / 96 + 0.02 * math.sin(j * 7.1)
                rr = r + (40 + 90 * ((j * 37) % 11) / 10) * ease_out(q)
                dp.setColor4f(rgba(255, 255, 255, 0.8 * (1 - q) ** 1.2 * fade))
                c.drawCircle(CX + rr * math.cos(a), CY + rr * math.sin(a), 2.2, dp)
        for sig, wdt, al, col in layers:
            pa = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=wdt,
                            StrokeCap=skia.Paint.kRound_Cap)
            pa.setColor4f(rgba(*col, al * fade))
            if sig:
                pa.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sig))
            c.drawArc(rect, start, sweep, False, pa)
        # label tag on top of each ring
        lab = RING_LABELS[k]
        tw = G["f_tag"].measureText(lab)
        tag = skia.RRect.MakeRectXY(skia.Rect(CX - tw / 2 - 12, CY - r - 17, CX + tw / 2 + 12, CY - r + 17), 6, 6)
        act = k == stage
        ta = (1.0 if act else 0.6 if k > stage else 1 - min(1.0, (t - G["ball"]["gaps"][k][2]) / 0.3)) * fade
        fp = skia.Paint(AntiAlias=True)
        fp.setColor4f(rgba(235, 240, 250, ta) if act else rgba(24, 31, 44, ta))
        c.drawRRect(tag, fp)
        if not act:
            sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.5)
            sp.setColor4f(rgba(120, 135, 160, ta * 0.8))
            c.drawRRect(tag, sp)
        tp = skia.Paint(AntiAlias=True)
        tp.setColor4f(rgba(12, 16, 24, ta) if act else rgba(170, 182, 200, ta))
        c.drawString(lab, CX - tw / 2, CY - r + 8, G["f_tag"], tp)


def draw_hits(c, t):
    skia = G["skia"]
    for tb, x, y, k, an in G["ball"]["hits"]:
        dt = t - tb
        if 0 <= dt < 0.35:
            a = (1 - dt / 0.35) ** 2
            r = RING_R[k]
            pa = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=10,
                            StrokeCap=skia.Paint.kRound_Cap,
                            MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 8))
            pa.setColor4f(rgba(*G["accent"], 0.9 * a))
            c.drawArc(skia.Rect(CX - r, CY - r, CX + r, CY + r), math.degrees(an) - 14, 28, False, pa)
            rp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2)
            rp.setColor4f(rgba(255, 255, 255, 0.6 * a))
            c.drawCircle(CX + x, CY + y, BALL_R + 40 * (dt / 0.35) ** 0.6, rp)


def draw_escape(c, t):
    skia = G["skia"]
    pp = skia.Paint(AntiAlias=True)
    for (T1, x, y, ang, k) in G["ball"]["escapes"]:
        dt = t - T1
        if not 0 <= dt < 1.5:
            continue
        if dt < 0.4:                                        # flash in the slit
            fl = skia.Paint(AntiAlias=True, MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 22))
            fl.setColor4f(rgba(*G["accent"], 0.9 * (1 - dt / 0.4)))
            c.drawCircle(CX + x, CY + y, 34 + 60 * dt, fl)
        for a0, vr, vt, life, sz in G["bursts"][k]:
            if dt > life:
                continue
            aa = ang + (a0 - math.pi) * 0.45
            sp = 90 + 2.2 * abs(vr)
            d = sp * dt * (1 - 0.35 * dt / life)
            pp.setColor4f(rgba(255, 255, 255, 0.9 * (1 - dt / life) ** 1.5))
            c.drawCircle(CX + x + d * math.cos(aa), CY + y + d * math.sin(aa), sz, pp)


def draw_ball(c, t, fade, pulse):
    skia = G["skia"]
    tr = skia.Paint(AntiAlias=True)
    for j in range(16, 0, -1):
        x, y = ball_pos(t - j * 0.011)
        tr.setColor4f(rgba(*G["accent"], 0.45 * (1 - j / 17) ** 1.6 * fade))
        c.drawCircle(x, y, BALL_R * (1 - j / 22), tr)
    x, y = ball_pos(t)
    gl = skia.Paint(AntiAlias=True, MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 14 + 8 * pulse))
    gl.setColor4f(rgba(*G["accent"], 0.85 * fade))
    c.drawCircle(x, y, BALL_R + 8 + 5 * pulse, gl)
    core = skia.Paint(AntiAlias=True)
    core.setColor4f(rgba(255, 255, 255, fade))
    c.drawCircle(x, y, BALL_R, core)


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
    cx, cy = CX, CY

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
        draw_rings(c, t, stage, 1.0, pulse)
        draw_hits(c, t)
        draw_escape(c, t)
        draw_ball(c, t, 1.0, pulse)
    else:
        # FULL SONG: radial spectrum
        dt = t - ts
        out = 1 - ease_out(dt / 0.7)
        if out > 0:
            draw_rings(c, t, 5, out, pulse)
            draw_escape(c, t)
            draw_ball(c, t, out, pulse)
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


X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


def render_segment(job):
    """Worker: render frames [a, b) and encode them straight to a video segment.
    Only the segment path goes back to the parent, never raw frames (Windows
    pipes refuse multi-MB results)."""
    a, b, path = job
    ff = subprocess.Popen(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", *X264, "-threads", "2", "-g", str(FPS * 2), path],
        stdin=subprocess.PIPE)
    for i in range(a, b):
        ff.stdin.write(render_frame(i))
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on segment {path}")
    return path


def render_video(audio, stage_times, out_beats, opts, out_path, workdir, progress=None):
    duration = len(audio) / SR
    n_frames = int(duration * FPS)
    pulse, level, spec = audio_features(audio, n_frames, stage_times[5])
    wav = os.path.join(workdir, "mix.wav")
    write_wav(wav, audio)
    o = dict(DEFAULT_OPTS, **{k: v for k, v in (opts or {}).items() if v is not None})
    accent = tuple(int(o["accent"].lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
    state = dict(stage_times=stage_times, duration=duration, pulse=pulse, level=level, spec=spec,
                 accent=accent, handle=o["handle"], credit=o["credit"],
                 cta=(o["cta"], o["cta2"]), ball=plan_ball(stage_times, out_beats))
    seg_dir = os.path.join(workdir, f"segments_{os.getpid()}_{int(time.time())}")
    os.makedirs(seg_dir, exist_ok=True)
    step = FPS * 4
    jobs = [(a, min(n_frames, a + step), os.path.join(seg_dir, f"seg_{k:04d}.mp4"))
            for k, a in enumerate(range(0, n_frames, step))]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ctx.Pool(workers, initializer=init_render, initargs=(state,)) as pool:
        for k, _ in enumerate(pool.imap_unordered(render_segment, jobs)):
            if progress:
                progress(0.97 * (k + 1) / len(jobs))
            log(f"render {k + 1}/{len(jobs)} segments ({time.time() - t0:.0f}s)")
    # stitch the segments (no re-encode) and add the audio
    lst = os.path.join(seg_dir, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for _, _, path in jobs:
            f.write(f"file '{os.path.basename(path)}'\n")
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", out_path], check=True)
    import shutil
    shutil.rmtree(seg_dir, ignore_errors=True)
    if progress:
        progress(1.0)
    log(f"video written: {out_path} ({duration:.1f}s, {time.time() - t0:.0f}s render)")


# ----------------------------------------------------------------------------- high level API
DEFAULT_OPTS = dict(handle="@yourhandle", credit="", accent="#9FB4FF",
                    cta="FOLLOW FOR THE NEXT DROP", cta2="new songs every day")


def load_song(song=None, stems_dir=None, workdir=".stem_reveal_cache", progress=None):
    """Decode + separate + analyse. Returns a context used by the other steps."""
    say = progress or (lambda stage, frac: None)
    say("decode", 0)
    if stems_dir:
        stems = load_stem_dir(stems_dir)
        mix = sum(stems.values())
        key = hashlib.md5(os.path.abspath(stems_dir).encode()).hexdigest()[:10]
    else:
        mix = decode(song)
        with open(song, "rb") as f:
            key = hashlib.md5(f.read(1 << 20) + str(len(mix)).encode()).hexdigest()[:10]
    work = os.path.join(workdir, key)
    os.makedirs(work, exist_ok=True)
    if not stems_dir:
        stems = separate(mix, work, progress=lambda f: say("separate", f))
        n = min(len(mix), *(len(v) for v in stems.values()))
        mix, stems = mix[:n], {k: v[:n] for k, v in stems.items()}
    say("analyze", 0)
    beats, bars, _ = analyze(mix, stems)
    feats = bar_features(mix, stems, bars)
    hook = find_hook(feats, sum(STAGE_BARS) // 2)
    log(f"hook starts at {bars[hook]:.2f}s (bar {hook})")
    return dict(mix=mix, stems=stems, beats=beats, bars=bars, feats=feats, auto_hook=hook, work=work, key=key)


def hook_bar_at(ctx, seconds):
    """Nearest bar start to a time in the song (manual choice snaps to the bar grid)."""
    return int(np.argmin(np.abs(ctx["bars"] - seconds)))


def build_audio(ctx, hook=None, min_duration=62.0, max_duration=None, stage_bars=None, full_bars=None):
    hook = ctx["auto_hook"] if hook is None else int(hook)
    hook = int(np.clip(hook, 0, len(ctx["bars"]) - 2))
    run = max(8, section_run(ctx["feats"], hook))
    audio, stage_times, out_beats, out_down, plan = arrange(
        ctx["mix"], ctx["stems"], ctx["beats"], ctx["bars"], hook, run, min_duration,
        max_duration, stage_bars, full_bars)
    plan["hook_bar"] = hook
    return audio, stage_times, out_beats, plan


def overview(ctx, points=1200):
    """Small summary of the song for the UI: envelope, bars, auto hook."""
    mix = ctx["mix"].mean(1)
    hop = max(1, len(mix) // points)
    env = np.sqrt((mix[: hop * points].reshape(points, hop) ** 2).mean(1))
    voc = ctx["stems"]["vocals"].mean(1)
    venv = np.sqrt((voc[: hop * points].reshape(points, hop) ** 2).mean(1))
    m = max(env.max(), 1e-9)
    return dict(duration=len(mix) / SR, bpm=round(60 / float(np.median(np.diff(ctx["beats"]))), 1),
                bars=[round(float(b), 3) for b in ctx["bars"]],
                env=[round(float(v / m), 3) for v in env], venv=[round(float(v / m), 3) for v in venv],
                auto_hook=int(ctx["auto_hook"]))


def make_video(ctx, out_path, hook=None, min_duration=62.0, opts=None, progress=None,
               max_duration=None, stage_bars=None, full_bars=None):
    audio, stage_times, out_beats, plan = build_audio(ctx, hook, min_duration, max_duration, stage_bars, full_bars)
    plan["out"] = out_path
    with open(os.path.splitext(out_path)[0] + ".plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    render_video(audio, stage_times, out_beats, opts, out_path, ctx["work"], progress=progress)
    return plan


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song", nargs="?", help="song file (mp3, wav, m4a, even a video)")
    ap.add_argument("--stems", help="folder with vocals/other/bass/drums files (skips separation)")
    ap.add_argument("-o", "--out", default="stem_reveal.mp4")
    ap.add_argument("--start", type=float, help="force the hook start time in seconds")
    ap.add_argument("--min-duration", type=float, default=62.0, help="minimum video length (s)")
    ap.add_argument("--max-duration", type=float, help="maximum video length (s)")
    ap.add_argument("--stage-bars", help='bars per build stage, e.g. "2,4,4,4,4" (muffled,vocals,melody,bass,drums)')
    ap.add_argument("--full-bars", type=int, help="bars of the FULL SONG part (default: auto)")
    ap.add_argument("--handle", default="@yourhandle")
    ap.add_argument("--credit", default="", help='e.g. "Song Title — Artist" (shown on FULL SONG)')
    ap.add_argument("--accent", default="#9FB4FF")
    ap.add_argument("--cta", default="FOLLOW FOR THE NEXT DROP")
    ap.add_argument("--cta2", default="new songs every day")
    ap.add_argument("--workdir", default=".stem_reveal_cache")
    args = ap.parse_args()
    if not args.song and not args.stems:
        ap.error("give a song file or --stems")

    ctx = load_song(args.song, args.stems, args.workdir)
    hook = hook_bar_at(ctx, args.start) if args.start is not None else None
    opts = dict(handle=args.handle, credit=args.credit, accent=args.accent, cta=args.cta, cta2=args.cta2)
    sb = [int(x) for x in args.stage_bars.split(",")] if args.stage_bars else None
    plan = make_video(ctx, args.out, hook, args.min_duration, opts,
                      max_duration=args.max_duration, stage_bars=sb, full_bars=args.full_bars)
    log("plan:", json.dumps(plan))


if __name__ == "__main__":
    main()
