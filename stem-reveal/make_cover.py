#!/usr/bin/env python3
"""
Cover / thumbnail (1080x1920) in the same style as the videos.

  python3 make_cover.py -o cover.png --line1 "GUESS" --line2 "THE SONG" --sub "5 layers. 1 drop."
  python3 make_cover.py -o cover.png --line1 "BLINDING" --line2 "LIGHTS" --sub "built layer by layer" --accent "#FF7AB6"

Everything important stays inside the centre 1080x1440 area (what the profile grid shows).
"""
import argparse
import math
import os

import numpy as np
import skia

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
W, H = 1080, 1920
CX, CY = W / 2, 1010
RINGS = [150, 225, 300, 375]
LABELS = ["VOCALS", "MELODY", "BASS", "DRUMS"]


def col(h, a=1.0):
    h = h.lstrip("#")
    return skia.Color4f(int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def blur(s):
    return skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, s)


def font(name, size):
    f = skia.Font(skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, name)), size)
    f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    return f


def fit(text, name, size, max_w):
    f = font(name, size)
    while f.measureText(text) > max_w and size > 40:
        size -= 4
        f = font(name, size)
    return f


def text_c(c, s, f, y, color, glow=None):
    x = CX - f.measureText(s) / 2
    if glow:
        g = skia.Paint(AntiAlias=True, MaskFilter=blur(18))
        g.setColor4f(glow)
        c.drawString(s, x, y, f, g)
    p = skia.Paint(AntiAlias=True)
    p.setColor4f(color)
    c.drawString(s, x, y, f, p)


def make(out, line1, line2, sub, accent, handle):
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=skia.GradientShader.MakeLinear(
        [skia.Point(0, 0), skia.Point(0, H)], [skia.Color(14, 19, 29), skia.Color(5, 7, 11)])))
    c.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=skia.GradientShader.MakeRadial(
        skia.Point(CX, CY), 820, [skia.Color(28, 38, 60, 200), skia.Color(0, 0, 0, 0)])))
    rng = np.random.default_rng(11)
    bp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2)
    for _ in range(60):
        bp.setColor4f(skia.Color4f(0.8, 0.86, 0.95, rng.uniform(0.06, 0.2)))
        c.drawCircle(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(4, 13), bp)

    # rings with their slits; the ball is escaping the outer one
    exit_ang = math.radians(-38)
    tag_f = font("Inter-Bold.otf", 24)
    for k, r in enumerate(RINGS):
        half = math.radians(11 - 1.8 * k)
        g = exit_ang + math.radians([95, -60, 40, 0][k])
        rect = skia.Rect(CX - r, CY - r, CX + r, CY + r)
        start, sweep = math.degrees(g + half), 360 - math.degrees(2 * half)
        outer = k == len(RINGS) - 1
        if outer:
            p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=26,
                           StrokeCap=skia.Paint.kRound_Cap, MaskFilter=blur(22))
            p.setColor4f(col(accent, 0.55))
            c.drawArc(rect, start, sweep, False, p)
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=[5, 6, 7, 10][k],
                       StrokeCap=skia.Paint.kRound_Cap)
        p.setColor4f(skia.Color4f(1, 1, 1, [0.28, 0.42, 0.6, 1.0][k]))
        c.drawArc(rect, start, sweep, False, p)
        # label tag at the top of the ring
        lab = LABELS[k]
        tw = tag_f.measureText(lab)
        tag = skia.RRect.MakeRectXY(skia.Rect(CX - tw / 2 - 13, CY - r - 19, CX + tw / 2 + 13, CY - r + 19), 7, 7)
        fp = skia.Paint(AntiAlias=True)
        fp.setColor4f(skia.Color4f(0.92, 0.94, 0.98, 1) if outer else skia.Color4f(0.10, 0.13, 0.19, 1))
        c.drawRRect(tag, fp)
        if not outer:
            sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.6)
            sp.setColor4f(skia.Color4f(0.47, 0.53, 0.63, 0.9))
            c.drawRRect(tag, sp)
        tp = skia.Paint(AntiAlias=True)
        tp.setColor4f(skia.Color4f(0.05, 0.06, 0.09, 1) if outer else skia.Color4f(0.67, 0.71, 0.78, 1))
        c.drawString(lab, CX - tw / 2, CY - r + 9, tag_f, tp)

    R = RINGS[-1]
    bx, by = CX + (R + 40) * math.cos(exit_ang), CY + (R + 40) * math.sin(exit_ang)
    d = np.array([math.cos(exit_ang + 0.12), math.sin(exit_ang + 0.12)])
    tp = skia.Paint(AntiAlias=True)
    for j in range(70, 0, -1):
        f = j / 70
        tp.setColor4f(col(accent, 0.5 * (1 - f) ** 1.6))
        c.drawCircle(bx - d[0] * 330 * f, by - d[1] * 330 * f, 26 * (1 - 0.75 * f), tp)
    sp = skia.Paint(AntiAlias=True)
    for _ in range(60):
        a = exit_ang + rng.normal(0, 0.6)
        dist = rng.uniform(25, 140)
        sp.setColor4f(skia.Color4f(1, 1, 1, rng.uniform(0.25, 0.9)))
        c.drawCircle(CX + R * math.cos(exit_ang) + dist * math.cos(a),
                     CY + R * math.sin(exit_ang) + dist * math.sin(a), rng.uniform(2, 5), sp)
    g = skia.Paint(AntiAlias=True, MaskFilter=blur(36))
    g.setColor4f(col(accent, 0.95))
    c.drawCircle(bx, by, 62, g)
    core = skia.Paint(AntiAlias=True)
    core.setColor4f(skia.Color4f(1, 1, 1, 1))
    c.drawCircle(bx, by, 26, core)

    # headline
    f1 = fit(line1, "InterDisplay-ExtraBold.otf", 150, W - 120)
    f2 = fit(line2, "InterDisplay-ExtraBold.otf", 150, W - 120)
    text_c(c, line1, f1, 430, skia.Color4f(1, 1, 1, 1))
    text_c(c, line2, f2, 430 + f2.getSize() * 0.98, col(accent), glow=col(accent, 0.55))
    if sub:
        text_c(c, sub, fit(sub, "Inter-SemiBold.otf", 50, W - 160), 1530, skia.Color4f(0.85, 0.88, 0.94, 0.95))
    # 0/5 -> 5/5 pill
    pill_f = font("Inter-Bold.otf", 38)
    pill = "0 / 5   →   5 / 5"
    pw = pill_f.measureText(pill)
    rr = skia.RRect.MakeRectXY(skia.Rect(CX - pw / 2 - 30, 1590, CX + pw / 2 + 30, 1660), 35, 35)
    pp = skia.Paint(AntiAlias=True)
    pp.setColor4f(col(accent, 0.16))
    c.drawRRect(rr, pp)
    ps = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2)
    ps.setColor4f(col(accent, 0.8))
    c.drawRRect(rr, ps)
    text_c(c, pill, pill_f, 1639, col(accent))
    if handle:
        text_c(c, handle, font("Inter-SemiBold.otf", 34), 230, skia.Color4f(0.6, 0.64, 0.72, 0.9))
    surf.makeImageSnapshot().save(out, skia.kPNG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="cover.png")
    ap.add_argument("--line1", default="GUESS")
    ap.add_argument("--line2", default="THE SONG")
    ap.add_argument("--sub", default="5 layers. 1 drop.")
    ap.add_argument("--accent", default="#9FB4FF")
    ap.add_argument("--handle", default="")
    a = ap.parse_args()
    make(a.out, a.line1.upper(), a.line2.upper(), a.sub, a.accent, a.handle)
    print("cover written:", a.out)
