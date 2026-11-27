"""
"Escape" mode — hundreds of thin rings around a small circle, a counter in the
middle. The ball bounces inside the circle: while it only hits the wall the
music is MUFFLED; when it lands in the rotating slit, rings break one after the
other, the counter drops and the music is CLEAR. At 0 the radial spectrum ends
the video.

Timing, all taken from the song:
  * every bounce lands on a beat,
  * clear moments start/stop on beats, the last beats of each 4-bar phrase,
    getting longer towards the hook,
  * the hook itself is fully clear and the counter reaches 0 exactly at its end.
"""
import math

import numpy as np

import stem_reveal as S

RC = 150.0                    # radius of the centre circle (the ball's cage)
SP = 4.6                      # spacing between rings (px)
MAX_R = 1170.0                # rings beyond this radius are off screen
HB = math.radians(24)         # half-width of the slit in the centre circle
OMEGA = math.radians(55)      # rotation speed of the slit (rad/s)
MARGIN = math.radians(16)     # a bounce never lands closer than this to the slit
BALL_R = 13
CX, CY = S.W / 2, 960

DEFAULTS = dict(rings=500, pre_bars=16, drop_bars=None, outro_bars=2)


def theta(t):
    return OMEGA * t - math.pi / 2


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- timeline + audio
def build_escape(ctx, hook=None, min_duration=62.0, max_duration=None, rings=500, pre_bars=16,
                 drop_bars=None, outro_bars=2):
    bars, beats, mix = ctx["bars"], ctx["beats"], ctx["mix"]
    nb = len(bars)
    hook = ctx["auto_hook"] if hook is None else int(hook)
    hook = int(np.clip(hook, 1, nb - 3))
    bar_dur = float(np.median(np.diff(bars)))
    song_end = len(mix) / S.SR

    def bt(i):
        return float(bars[i]) if i < nb else float(bars[-1] + (i - nb + 1) * bar_dur)

    pre = int(np.clip(int(pre_bars or 16), 4, hook))
    out_b = max(1, int(outro_bars))

    def total(pre_, F_):
        return min(bt(hook + F_ + out_b), song_end) - bt(hook - pre_)

    if drop_bars:
        F = max(1, int(drop_bars))
    else:
        F = 4
        while total(pre, F) < min_duration and hook + F + out_b < nb - 1:
            F += 4
    F = max(1, min(F, nb - 1 - hook - out_b))
    if max_duration:
        while F > 2 and total(pre, F) > max_duration + 1e-6:
            F -= 1
        while pre > 4 and total(pre, F) > max_duration + 1e-6:
            pre -= 4
        if total(pre, F) > max_duration + 1e-6:
            raise ValueError(f"impossible de tenir en {max_duration:.0f} s avec ce son : il faut au moins "
                             f"{total(pre, F):.1f} s. Augmente la durée max.")

    s0 = bt(hook - pre)
    end = min(bt(hook + F + out_b), song_end)
    finale = bt(hook + F) - s0
    beats_out = np.array([b - s0 for b in beats if s0 - 1e-3 <= b < end - 1e-3])

    # ---- muffled / clear phases (output time)
    phases = []
    first_full = hook - pre + (pre % 4)
    if pre % 4:
        phases.append([0.0, bt(first_full) - s0, "M"])
    n_phr = (hook - first_full) // 4
    for j in range(n_phr):
        b0 = first_full + 4 * j
        t0, t1 = bt(b0) - s0, bt(b0 + 4) - s0
        clear_beats = min(12, 4 * (1 + (2 * j) // max(1, n_phr - 1)))
        if j == 0 and n_phr > 1:
            clear_beats = 2                          # the first escape is a short tease
        inner = [b - s0 for b in beats if bt(b0) - 1e-3 <= b < bt(b0 + 4) - 1e-3]
        k = max(1, len(inner) - clear_beats)
        tc = inner[k] if k < len(inner) else t1 - (t1 - t0) * clear_beats / 16
        phases.append([t0, tc, "M"])
        phases.append([tc, t1, "C"])
    phases.append([bt(hook) - s0, finale, "C"])
    merged = []
    for p in phases:
        if p[1] - p[0] < 1e-3:
            continue
        if merged and merged[-1][2] == p[2] and abs(merged[-1][1] - p[0]) < 1e-3:
            merged[-1][1] = p[1]
        else:
            merged.append(list(p))
    phases = merged

    # ---- audio: one continuous take of the song, low-passed while the ball is stuck
    a, b = int(round(s0 * S.SR)), int(round(end * S.SR))
    clear = mix[a:b].astype(np.float32)
    muff = S.lowpass(clear, 420)
    rc = np.sqrt((clear ** 2).mean() + 1e-12)
    rm = np.sqrt((muff ** 2).mean() + 1e-12)
    muff *= float(np.clip(rc / rm * 10 ** (-3 / 20), 1, 10 ** (9 / 20)))
    w = np.zeros(len(clear), np.float32)
    for t0, t1, kind in phases:
        if kind == "C":
            w[int(t0 * S.SR): int(t1 * S.SR)] = 1
    w[int(finale * S.SR):] = 1
    ramp = int(0.025 * S.SR)
    kern = np.ones(ramp, np.float32) / ramp
    w = np.convolve(w, kern)[: len(w)]                # 25 ms fade, starting on the beat
    audio = w[:, None] * clear + (1 - w[:, None]) * muff
    nf = int(1.6 * S.SR)
    audio[-nf:] *= np.linspace(1, 0, nf, dtype=np.float32)[:, None] ** 1.5
    peak = np.percentile(np.abs(audio), 99.995)
    audio = audio * (0.92 / max(peak, 1e-6))
    audio = np.where(np.abs(audio) > 0.9, np.sign(audio) * (0.9 + 0.1 * np.tanh((np.abs(audio) - 0.9) / 0.1)), audio)

    plan = dict(mode="escape", bpm=round(60 / float(np.median(np.diff(beats))), 1), hook_bar=hook,
                hook_start_s=round(bt(hook), 2), segment_start_s=round(s0, 2), pre_bars=pre, drop_bars=F,
                outro_bars=out_b, rings=int(rings), bar_s=round(bar_dur, 3),
                phases=[[round(p[0], 3), round(p[1], 3), p[2]] for p in phases],
                finale_s=round(finale, 3), duration_s=round(len(audio) / S.SR, 2))
    return audio.astype(np.float32), beats_out, plan


# ----------------------------------------------------------------------------- ball + rings plan
def plan_escape(plan, beats_out):
    phases, finale, N = plan["phases"], plan["finale_s"], plan["rings"]
    rng = np.random.default_rng(5)
    Rb = RC - BALL_R - 3
    segs = []                          # ("line", t0, t1, p0, p1) / ("ride", t0, t1)
    hits = []
    cur_t, cur_p, cur_ang = 0.0, np.zeros(2), None
    cands = [math.radians(d) for d in range(95, 170, 6)]

    def solve(times, final_ang, start_ang):
        n = len(times)
        angs = [None] * n
        angs[-1] = final_ang
        budget = [20000]

        def ok(a, t):
            return abs(wrap(a - theta(t))) > HB + MARGIN

        def rec(k):
            if k < 0:
                return True
            budget[0] -= 1
            if budget[0] < 0:
                return False
            for o in rng.permutation(2 * len(cands)):
                d = cands[o // 2] * (1 if o % 2 else -1)
                x = angs[k + 1] - d
                if not ok(x, times[k]):
                    continue
                if k == 0 and start_ang is not None and abs(wrap(x - start_ang)) < math.radians(60):
                    continue
                angs[k] = x
                if rec(k - 1):
                    return True
            return False
        if not rec(n - 2):                               # very unlikely: fall back to plain steps
            for k in range(n - 2, -1, -1):
                angs[k] = angs[k + 1] - math.radians(137)
        return angs

    for t0, t1, kind in phases:
        if kind == "M":
            times = [float(b) for b in beats_out if t0 + 0.08 < b < t1 - 0.08] + [t1]
            angs = solve(times, theta(t1), cur_ang)
            for k, (tk, ak) in enumerate(zip(times, angs)):
                last = k == len(times) - 1
                r = RC if last else Rb
                p = np.array([r * math.cos(ak), r * math.sin(ak)])
                segs.append(("line", cur_t, tk, cur_p, p))
                if not last:
                    hits.append((tk, ak))
                cur_t, cur_p = tk, p
            cur_ang = theta(t1)
        else:
            if cur_t < t0 - 1e-6:                        # (only if the video starts clear)
                p = np.array([RC * math.cos(theta(t0)), RC * math.sin(theta(t0))])
                segs.append(("line", cur_t, t0, cur_p, p))
            segs.append(("ride", t0, t1))
            cur_t = t1
            cur_p = np.array([RC * math.cos(theta(t1)), RC * math.sin(theta(t1))])
            cur_ang = theta(t1)
    # after the last ring: the ball flies away
    d = cur_p / (np.linalg.norm(cur_p) + 1e-9)
    segs.append(("line", cur_t, cur_t + 1.6, cur_p, cur_p + d * 1400))

    # rings broken over time: constant rate during the clear moments, 0 left at the finale
    clear_total = sum(t1 - t0 for t0, t1, k in phases if k == "C")
    kt, kd, acc = [0.0], [0.0], 0.0
    for t0, t1, kind in phases:
        if kind == "C":
            kt += [t0, t1]
            kd += [acc, acc + N * (t1 - t0) / clear_total]
            acc = kd[-1]
        else:
            kt += [t0, t1]
            kd += [acc, acc]
    kt.append(finale + 100)
    kd.append(float(N))
    return dict(segs=segs, hits=hits, kt=np.array(kt), kd=np.array(kd))


# ----------------------------------------------------------------------------- rendering (worker side)
def ball_pos(t):
    for seg in S.G["esc"]["segs"]:
        if t <= seg[2] or seg is S.G["esc"]["segs"][-1]:
            if seg[0] == "ride":
                a = theta(t)
                return CX + RC * math.cos(a), CY + RC * math.sin(a)
            _, t0, t1, p0, p1 = seg
            u = min(1.0, max(0.0, (t - t0) / max(1e-6, t1 - t0)))
            return CX + p0[0] + (p1[0] - p0[0]) * u, CY + p0[1] + (p1[1] - p0[1]) * u
    return CX, CY


def broken(t):
    E = S.G["esc"]
    return float(np.interp(t, E["kt"], E["kd"]))


def is_clear(t):
    for t0, t1, kind in S.G["phases"]:
        if t0 <= t < t1:
            return kind == "C"
    return t >= S.G["finale"]


def ring_style(i):
    """Colour bands (accent / dark / light) and jagged slit width for ring #i."""
    cache = S.G.setdefault("_ring_cache", {})
    if i in cache:
        return cache[i]
    ar, ag, ab = S.G["accent"]
    ph = (i % 38) / 38
    k = 0.5 + 0.5 * math.cos(2 * math.pi * ph)            # 1 = light band, 0 = dark band
    light = (ar + (255 - ar) * 0.55, ag + (255 - ag) * 0.55, ab + (255 - ab) * 0.55)
    dark = (ar * 0.42, ag * 0.42, ab * 0.42)
    m = 0.5 + 0.5 * math.sin(2 * math.pi * ph * 2)
    base = tuple(dark[c] + (S.G["accent"][c] - dark[c]) * m for c in range(3))
    col = tuple(base[c] + (light[c] - base[c]) * k * 0.7 for c in range(3))
    jag = math.radians(9) * (math.sin(i * 0.37) + 0.6 * math.sin(i * 1.13 + 1.7) + 0.35 * math.sin(i * 2.9)) \
        + math.radians(7) * ((i * 7919) % 13) / 13
    cache[i] = (col, HB * 0.95 + abs(jag))
    return cache[i]


def render_frame(i):
    G = S.G
    skia = G["skia"]
    t = i / S.FPS
    pulse = float(G["pulse"][min(i, len(G["pulse"]) - 1)])
    finale = G["finale"]
    N = G["rings"]
    rgba = S.rgba
    surf = G["surface"]
    c = surf.getCanvas()
    ar, ag, ab = G["accent"]
    c.clear(skia.Color(int(ar * 0.05) + 6, int(ag * 0.05) + 5, int(ab * 0.05) + 7))
    d = broken(t)
    clear_now = is_clear(t)

    if t < finale + 0.8:
        fade = 1 - S.ease_out((t - finale) / 0.8) if t > finale else 1.0
        th = theta(t)
        first = int(math.floor(d)) + 1
        last = min(N - 1, first + int((MAX_R - RC) / SP) + 2)
        pa = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=SP * 0.72)
        for k in range(first, last + 1):
            r = RC + (k - d) * SP
            if r < RC + 2:
                continue
            col, half = ring_style(k)
            centre = th + 0.0011 * (r - RC)                   # slight spiral
            pa.setColor4f(rgba(*col, 0.95 * fade))
            c.drawArc(skia.Rect(CX - r, CY - r, CX + r, CY + r), math.degrees(centre + half),
                      360 - math.degrees(2 * half), False, pa)
        # centre circle with its slit
        c.drawCircle(CX, CY, RC - 2, skia.Paint(AntiAlias=True, Color=skia.Color(8, 6, 9)))
        hit_glow = 0.0
        for tk, ak in G["esc"]["hits"]:
            if 0 <= t - tk < 0.3:
                hit_glow = max(hit_glow, 1 - (t - tk) / 0.3)
        for sig, wdt, al, col in ((14, 12, (0.35 + 0.5 * hit_glow) * fade, (ar, ag, ab)),
                                  (0, 4.5, 0.95 * fade, (255, 255, 255))):
            p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=wdt,
                           StrokeCap=skia.Paint.kRound_Cap)
            p.setColor4f(rgba(*col, al))
            if sig:
                p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sig))
            c.drawArc(skia.Rect(CX - RC, CY - RC, CX + RC, CY + RC), math.degrees(th + HB),
                      360 - math.degrees(2 * HB), False, p)
        # bounce flashes
        for tk, ak in G["esc"]["hits"]:
            dt = t - tk
            if 0 <= dt < 0.3:
                q = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2)
                q.setColor4f(rgba(255, 255, 255, 0.6 * (1 - dt / 0.3)))
                c.drawCircle(CX + (RC - BALL_R - 3) * math.cos(ak), CY + (RC - BALL_R - 3) * math.sin(ak),
                             BALL_R + 34 * (dt / 0.3) ** 0.6, q)
        # sparks flying out of the slit while rings break
        sp = skia.Paint(AntiAlias=True)
        for j in range(36):
            tau = t - j / 60
            if tau < 0 or not is_clear(tau) or tau > finale:
                continue
            age = t - tau
            a0 = theta(tau) + 0.5 * math.sin(j * 12.9898 + tau * 7.0)
            dist = RC + 30 + 900 * age * (0.6 + 0.4 * math.sin(j * 4.1 + 1.3) ** 2)
            sp.setColor4f(rgba(255, 255, 255, 0.85 * (1 - age / 0.6)))
            c.drawCircle(CX + dist * math.cos(a0), CY + dist * math.sin(a0), 2.6, sp)
        # counter
        left = max(0, N - int(math.floor(d + 1e-6)))
        tp = skia.Paint(AntiAlias=True)
        tp.setColor4f(rgba(255, 255, 255, fade))
        f = G["f_counter"]
        txt = str(left)
        c.drawString(txt, CX - f.measureText(txt) / 2, CY + 27, f, tp)
        # ball
        tr = skia.Paint(AntiAlias=True)
        for j in range(14, 0, -1):
            x, y = ball_pos(t - j * 0.012)
            tr.setColor4f(rgba(ar, ag, ab, 0.4 * (1 - j / 15) ** 1.6 * fade))
            c.drawCircle(x, y, BALL_R * (1 - j / 20), tr)
        x, y = ball_pos(t)
        gl = skia.Paint(AntiAlias=True, MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 14 + 10 * clear_now))
        gl.setColor4f(rgba(ar, ag, ab, 0.9 * fade))
        c.drawCircle(x, y, BALL_R + 8 + 6 * clear_now, gl)
        core = skia.Paint(AntiAlias=True)
        core.setColor4f(rgba(255, 255, 255, fade))
        c.drawCircle(x, y, BALL_R, core)
        # muffled = slightly darker picture (eases in/out in 0.12 s)
        dim = 0.0 if clear_now else 0.22
        for back in (0.04, 0.08, 0.12):
            if is_clear(t - back) != clear_now:
                dim *= back / 0.12
                break
        if dim > 0:
            ov = skia.Paint()
            ov.setColor4f(rgba(0, 0, 0, dim))
            c.drawRect(skia.Rect(0, 0, S.W, S.H), ov)

    if t >= finale:
        S.draw_finale(c, i, t - finale, pulse, CX, CY)
        if G["credit"]:
            cr = skia.Paint(AntiAlias=True)
            cr.setColor4f(rgba(170, 178, 192, S.ease_out((t - finale - 0.6) / 0.6)))
            S.text_c(c, G["credit"], G["f_small"], CX, 430, cr)

    if G["handle"]:
        hp = skia.Paint(AntiAlias=True)
        hp.setColor4f(rgba(160, 168, 182, 0.8))
        S.text_c(c, G["handle"], G["f_small"], CX, 228, hp)
    fade = min(1.0, t / 0.25, (G["duration"] - t) / 1.2)
    if fade < 1:
        fp = skia.Paint()
        fp.setColor4f(rgba(0, 0, 0, 1 - max(0.0, fade)))
        c.drawRect(skia.Rect(0, 0, S.W, S.H), fp)
    return surf.makeImageSnapshot().toarray().tobytes()


# ----------------------------------------------------------------------------- entry point
def make_escape_video(ctx, out_path, hook=None, opts=None, progress=None, **settings):
    import json
    import os
    audio, beats_out, plan = build_escape(ctx, hook, **settings)
    plan["out"] = out_path
    with open(os.path.splitext(out_path)[0] + ".plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    duration = len(audio) / S.SR
    n_frames = int(duration * S.FPS)
    pulse, level, spec = S.audio_features(audio, n_frames, plan["finale_s"])
    wav = os.path.join(ctx["work"], "mix_escape.wav")
    S.write_wav(wav, audio)
    o = dict(S.DEFAULT_OPTS, **{k: v for k, v in (opts or {}).items() if v is not None})
    state = dict(mode="escape", duration=duration, pulse=pulse, level=level, spec=spec,
                 accent=S.hex_rgb(o["accent"]), handle=o["handle"], credit=o["credit"],
                 cta=(o["cta"], o["cta2"]), esc=plan_escape(plan, beats_out),
                 phases=plan["phases"], finale=plan["finale_s"], rings=plan["rings"])
    S.encode_frames(state, n_frames, wav, out_path, ctx["work"], progress)
    return plan
