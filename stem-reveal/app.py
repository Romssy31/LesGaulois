#!/usr/bin/env python3
"""
Stem Reveal — local app.  Run:  python3 app.py   (opens http://127.0.0.1:7860)

Drop a song, it separates the stems, finds tempo / bars / the hook, lets you
check or move the hook on the waveform, previews the audio edit and renders
the video.
"""
import html
import os
import secrets
import socket
import sys
import threading
import time
import traceback
import uuid
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402

import escape_mode as ESC  # noqa: E402
import stem_reveal as SR  # noqa: E402

DATA = os.path.join(HERE, "projects")
UPLOADS = os.path.join(DATA, "uploads")
OUTPUT = os.path.join(DATA, "videos")
CACHE = os.path.join(DATA, "cache")
for d in (UPLOADS, OUTPUT, CACHE):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
SONGS = {}           # song_id -> analysis context (kept in memory)
SHARES = {}          # token -> video file name (phone download links)
PORT = int(os.environ.get("STEM_REVEAL_PORT", "7860"))
JOBS = {}            # job_id -> {state, stage, progress, message, result}
HEAVY = threading.Lock()   # one separation / render at a time


def start_job(fn, *a):
    jid = uuid.uuid4().hex[:12]
    job = JOBS[jid] = dict(state="queued", stage="", progress=0.0, message="", result=None, t0=time.time())

    def run():
        with HEAVY:
            job["state"] = "running"
            try:
                job["result"] = fn(job, *a)
                job["state"] = "done"
                job["progress"] = 1.0
            except Exception as e:                 # surfaced in the UI
                traceback.print_exc()
                job["state"] = "error"
                job["message"] = f"{type(e).__name__}: {e}"[:400]
    threading.Thread(target=run, daemon=True).start()
    return jid


# ----------------------------------------------------------------------------- jobs
def analyse_job(job, path, name):
    weights = dict(decode=(0.0, 0.05), separate=(0.05, 0.85), analyze=(0.85, 1.0))

    def progress(stage, frac):
        a, b = weights.get(stage, (0, 1))
        job["stage"] = stage
        job["progress"] = a + (b - a) * frac
    ctx = SR.load_song(path, workdir=CACHE, progress=progress)
    sid = uuid.uuid4().hex[:10]
    ctx["name"] = name
    ctx["path"] = path
    SONGS[sid] = ctx
    return dict(song_id=sid, name=name, audio_url="/files/uploads/" + os.path.basename(path), **SR.overview(ctx))


def timing_args(p):
    """Duration settings sent by the UI."""
    sb = p.get("stage_bars")
    mx = p.get("max_duration")
    fb = p.get("full_bars")
    rg = p.get("range")
    return dict(range_bars=[int(rg[0]), int(rg[1])] if rg else None,
                min_duration=float(p.get("min_duration") or 62),
                max_duration=float(mx) if mx not in (None, "", 0) else None,
                stage_bars=[int(x) for x in sb] if sb else None,
                full_bars=int(fb) if fb not in (None, "", 0) else None)


def escape_args(p):
    t = timing_args(p)
    db = p.get("drop_bars")
    return dict(range_bars=t["range_bars"], min_duration=t["min_duration"], max_duration=t["max_duration"],
                rings=int(p.get("rings") or 500), pre_bars=int(p.get("pre_bars") or 16),
                drop_bars=int(db) if db not in (None, "", 0) else None,
                pattern=str(p["pattern"]) if p.get("pattern") else None)


def render_job(job, sid, hook, timing, opts, mode="layers"):
    ctx = SONGS[sid]
    base = os.path.splitext(ctx["name"])[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in base).strip() or "video"
    fname = f"{safe} - stem reveal {time.strftime('%H%M%S')}.mp4"
    out = os.path.join(OUTPUT, fname)
    job["stage"] = "render"

    def progress(f):
        job["progress"] = 0.03 + 0.95 * f
    if mode == "escape":
        plan = ESC.make_escape_video(ctx, out, hook, opts, progress=progress, **timing)
        return finished(fname, plan)
    plan = SR.make_video(ctx, out, hook, timing["min_duration"], opts, progress=progress,
                         max_duration=timing["max_duration"], stage_bars=timing["stage_bars"],
                         full_bars=timing["full_bars"], range_bars=timing["range_bars"])
    return finished(fname, plan)


def finished(fname, plan):
    token = secrets.token_urlsafe(9)
    SHARES[token] = fname
    return dict(video_url="/files/videos/" + fname, filename=fname, plan=plan, share=token,
                phone_url=f"http://{lan_ip()}:{PORT}/m/{token}")


# ----------------------------------------------------------------------------- phone download (same Wi-Fi)
def lan_ip():
    """This computer's address on the local network (what the phone must open)."""
    s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s_.connect(("10.255.255.255", 1))          # no packet is sent, it only picks the right interface
        return s_.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s_.close()


@app.before_request
def only_phone_pages_from_network():
    """Other devices on the Wi-Fi only get the phone page of a finished video (with its secret
    token) and that video file. Everything else stays reachable from this computer only."""
    if request.remote_addr in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        return None
    path = request.path
    if path.startswith("/m/") and path[3:] in SHARES:
        return None
    if path.startswith("/files/videos/") and SHARES.get(request.args.get("t")) == path[len("/files/videos/"):]:
        return None
    abort(403)


@app.get("/api/qr/<token>")
def qr(token):
    if token not in SHARES:
        abort(404)
    url = f"http://{lan_ip()}:{PORT}/m/{token}"
    try:
        import segno
        svg = segno.make(url, error="m").svg_inline(scale=6, border=2, dark="#0b0f18", light="#ffffff")
    except ImportError:
        svg = ""
    return jsonify(url=url, svg=svg)


@app.get("/m/<token>")
def phone_page(token):
    name = SHARES.get(token) or abort(404)
    src = f"/files/videos/{name}?t={token}"
    return PHONE_PAGE.replace("{{SRC}}", html.escape(src, quote=True)).replace("{{NAME}}", html.escape(name))


PHONE_PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stem Reveal</title>
<style>
 body{margin:0;background:#070a10;color:#e8edf6;font:16px/1.45 -apple-system,system-ui,sans-serif;padding:16px}
 h1{font-size:20px;margin:4px 0 12px} video{width:100%;border-radius:14px;background:#000;max-height:60vh}
 .btn{display:block;width:100%;box-sizing:border-box;margin-top:14px;padding:15px;border:0;border-radius:12px;
      font:700 17px -apple-system,system-ui;text-align:center;text-decoration:none;background:#9fb4ff;color:#0b0f18}
 .alt{background:#151c28;color:#e8edf6;border:1px solid #222b3a}
 ol{color:#8b96a8;font-size:15px;padding-left:20px;margin-top:18px} li{margin-bottom:8px} b{color:#e8edf6}
 #msg{color:#7fe0a8;font-size:14px;min-height:20px;margin-top:8px}
 [hidden]{display:none!important}
</style></head><body>
<h1>Ta vidéo est prête</h1>
<video src="{{SRC}}" controls playsinline preload="metadata"></video>
<button class="btn" id="save" hidden>Enregistrer dans Photos</button>
<a class="btn" id="dl" href="{{SRC}}&dl=1" download="{{NAME}}">Télécharger la vidéo</a>
<div id="msg"></div>
<ol id="steps">
 <li>Touche <b>Télécharger la vidéo</b>, puis <b>Télécharger</b> si Safari le demande.</li>
 <li>Ouvre l'app <b>Fichiers</b> → <b>Téléchargements</b> et touche la vidéo.</li>
 <li>Touche <b>Partager</b> (le carré avec la flèche) → <b>Enregistrer la vidéo</b> : elle arrive dans <b>Photos</b>.</li>
 <li>Dans TikTok : <b>+</b> → <b>Importer</b> et choisis la vidéo.</li>
</ol>
<script>
// Direct "save to Photos" (share sheet with the file) is only allowed on secure pages.
const msg = document.getElementById("msg"), save = document.getElementById("save");
if (window.isSecureContext && navigator.canShare) {
  save.hidden = false; document.getElementById("dl").className = "btn alt";
  save.onclick = async () => {
    try {
      msg.textContent = "Préparation de la vidéo…";
      const blob = await (await fetch("{{SRC}}")).blob();
      const file = new File([blob], "{{NAME}}", { type: "video/mp4" });
      if (navigator.canShare({ files: [file] })) { await navigator.share({ files: [file] }); msg.textContent = ""; }
      else msg.textContent = "Utilise « Télécharger la vidéo ».";
    } catch (e) { msg.textContent = e.name === "AbortError" ? "" : "Utilise « Télécharger la vidéo »."; }
  };
}
</script></body></html>"""


# ----------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/song")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, "no file")
    ext = os.path.splitext(f.filename)[1].lower() or ".mp3"
    path = os.path.join(UPLOADS, uuid.uuid4().hex[:10] + ext)
    f.save(path)
    return jsonify(job=start_job(analyse_job, path, f.filename))


@app.get("/api/job/<jid>")
def job_status(jid):
    j = JOBS.get(jid) or abort(404)
    return jsonify({k: v for k, v in j.items() if k != "t0"} | dict(elapsed=round(time.time() - j["t0"])))


@app.post("/api/preview")
def preview():
    p = request.get_json(force=True)
    ctx = SONGS.get(p.get("song_id")) or abort(404, "song not loaded (re-import it)")
    try:
        if p.get("mode") == "escape":
            audio, out_beats, plan = ESC.build_escape(ctx, p.get("hook_bar"), **escape_args(p))
        else:
            audio, stage_times, out_beats, plan = SR.build_audio(ctx, p.get("hook_bar"), **timing_args(p))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    fname = f"preview_{p['song_id']}_{plan['hook_bar']}_{int(time.time())}.wav"
    SR.write_wav(os.path.join(OUTPUT, fname), audio)
    return jsonify(audio_url="/files/videos/" + fname, plan=plan,
                   beats=[round(float(b), 3) for b in out_beats])


@app.post("/api/render")
def render():
    p = request.get_json(force=True)
    sid = p.get("song_id")
    if sid not in SONGS:
        abort(404, "song not loaded (re-import it)")
    opts = {k: p.get(k) for k in ("handle", "credit", "accent", "cta", "cta2")}
    mode = "escape" if p.get("mode") == "escape" else "layers"
    t = escape_args(p) if mode == "escape" else timing_args(p)
    try:                                                    # validate before starting a long job
        if mode == "escape":
            ESC.build_escape(SONGS[sid], p.get("hook_bar"), **t)
        else:
            SR.build_audio(SONGS[sid], p.get("hook_bar"), **t)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(job=start_job(render_job, sid, p.get("hook_bar"), t, opts, mode))


@app.get("/files/<sub>/<path:name>")
def files(sub, name):
    root = {"uploads": UPLOADS, "videos": OUTPUT}.get(sub) or abort(404)
    return send_from_directory(root, name, as_attachment=request.args.get("dl") == "1")


def main():
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Stem Reveal is running: {url}\n  (close this window to stop it)\n"
          f"  Phone download links use http://{lan_ip()}:{PORT} (same Wi-Fi)\n", flush=True)
    if os.environ.get("STEM_REVEAL_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    # listen on the local network too, so the phone can fetch finished videos (see before_request)
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
