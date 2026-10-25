#!/usr/bin/env python3
"""Build a DASH ladder (2 s segments, fMP4, init + <name>N.m4s) + MPD in the SAME layout as the
existing bunny_2s_1080p tree (custom dash-js player, SegmentList).

Rungs are given as  <kbps>[@<height>]  e.g.  4000  5000@1440  12000@2160  40000@4320   (default height 1080).
Frames are 16:9 (1920x1080, 2560x1440, 3840x2160, 7680x4320): the 2.24:1 film is scaled to the full width and
letterboxed with black bars.  NOTE the 4K master is 3840 wide, so 4320p rungs are UPSCALED (no extra detail).

  1080p input : 1080p mezzanine (made once from the master)
  other       : the master itself
  encoder     : x264 CBR (nal-hrd=cbr -> real bytes == nominal bitrate), High profile, closed GOP 48, no scenecut
  mpd         : <prefix>_ibmff_mixed_<tag>.mpd (absolute BaseURL) and ...-public-access.mpd (relative BaseURL)

Usage:
  make_ladder.py --src SRC --out OUTROOT --workdir WORK [--start S --dur D] [--rungs ...] [--batch 4]
"""
import argparse
import os
import subprocess
import sys

PREFIX = "tos"
DEFAULT_RUNGS = "100,200,300,500,800,1000,1250,1500,1800,2000,2500,3000,3500,4000," \
                "5000@1440,6000@1440,7000@1440,8000@1440," \
                "10000@2160,12000@2160,16000@2160,20000@2160," \
                "30000@4320,40000@4320"
SEG = 2
# height -> (width, x264 level, MPD codec string)
RES = {
    1080: (1920, "4.0", "avc3.640028"),
    1440: (2560, "5.0", "avc3.640032"),
    2160: (3840, "5.1", "avc3.640033"),
    4320: (7680, "6.0", "avc3.64003c"),
}
PRESET = {1080: "fast", 1440: "veryfast", 2160: "veryfast", 4320: "ultrafast"}


def parse_rungs(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if "@" in tok:
            r, h = tok.split("@")
            out.append((int(r), int(h)))
        else:
            out.append((int(tok), 1080))
    return out


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def vf(h):
    w = RES[h][0]
    # scale to full width, letterbox to 16:9, make the pad offset even
    return f"scale={w}:-2:flags=lanczos,pad={w}:{h}:0:trunc((oh-ih)/4)*2:black,setsar=1,format=yuv420p"


def make_mezz(src, mezz, start, dur):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-an", "-map", "0:v:0", "-vf", vf(1080),
            "-r", "24", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
            "-x264-params", "keyint=48:min-keyint=48:scenecut=0", mezz]
    run(cmd)


def encode_batch(inp, start, dur, rootdir, rungs, h):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if inp_is_master(inp, h) and start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", inp]
    if inp_is_master(inp, h) and dur:
        cmd += ["-t", str(dur)]
    for r in rungs:
        d = os.path.join(rootdir, f"{PREFIX}_2s_{r}kbit")
        os.makedirs(d, exist_ok=True)
        rate = f"{r}k"
        buf = f"{2 * r}k"
        _, level, _ = RES[h]
        cmd += ["-map", "0:v:0", "-vf", vf(h) if h != 1080 else "format=yuv420p",
                "-r", "24", "-c:v", "libx264", "-preset", PRESET[h], "-profile:v", "high", "-level", level,
                "-b:v", rate, "-minrate", rate, "-maxrate", rate, "-bufsize", buf,
                "-x264-params", "keyint=48:min-keyint=48:scenecut=0:nal-hrd=cbr:force-cfr=1",
                "-pix_fmt", "yuv420p", "-an",
                "-f", "dash", "-seg_duration", str(SEG), "-use_template", "0", "-use_timeline", "0",
                "-dash_segment_type", "mp4", "-streaming", "0", "-remove_at_exit", "0",
                "-init_seg_name", f"{PREFIX}_2sinit.mp4", "-media_seg_name", f"{PREFIX}_2s$Number$.m4s",
                os.path.join(d, "manifest.mpd")]
    run(cmd)


def inp_is_master(inp, h):
    return h != 1080


def rung_stats(rootdir, r):
    d = os.path.join(rootdir, f"{PREFIX}_2s_{r}kbit")
    n = 0
    total = 0
    while os.path.exists(os.path.join(d, f"{PREFIX}_2s{n + 1}.m4s")):
        total += os.path.getsize(os.path.join(d, f"{PREFIX}_2s{n + 1}.m4s"))
        n += 1
    return n, total


def fmt_dur(sec):
    m = int(sec // 60)
    s = sec - 60 * m
    return f"PT0H{m}M{s:.2f}S"


def write_mpd(rootdir, mpd_path, base_url, rungs):
    stats = {r: rung_stats(rootdir, r) for r, _ in rungs}
    nseg = min(n for n, _ in stats.values())
    dur = nseg * SEG
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<MPD xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    out.append('     xmlns="urn:mpeg:DASH:schema:MPD:2011"')
    out.append('     xsi:schemaLocation="urn:mpeg:DASH:schema:MPD:2011"')
    out.append('     profiles="urn:mpeg:dash:profile:isoff-main:2011"')
    out.append('     type="static"')
    out.append(f'     mediaPresentationDuration="{fmt_dur(dur)}"')
    out.append('     minBufferTime="PT2.0S">   ')
    out.append(f'     <BaseURL>{base_url}</BaseURL>')
    out.append('     <Period start="PT0S">')
    out.append('        <AdaptationSet bitstreamSwitching="true">')
    for rid, (r, h) in enumerate(sorted(rungs)):
        n, total = stats[r]
        bw = int(total * 8 / (n * SEG))
        w, _, codec = RES[h]
        out.append(f'           <Representation id="{rid}" codecs="{codec}" mimeType="video/mp4" width="{w}" '
                   f'height="{h}" startWithSAP="1" bandwidth="{bw}">')
        out.append('             <SegmentBase>')
        out.append(f'               <Initialization sourceURL="{PREFIX}_2s_{r}kbit/{PREFIX}_2sinit.mp4"/>')
        out.append('                 </SegmentBase>')
        out.append(f'                 <SegmentList duration="{SEG}">')
        for k in range(1, nseg + 1):
            out.append(f'                    <SegmentURL media="{PREFIX}_2s_{r}kbit/{PREFIX}_2s{k}.m4s"/>')
        out.append('                </SegmentList>')
        out.append('              </Representation>')
        out.append('')
    out.append('          </AdaptationSet>')
    out.append('    </Period>')
    out.append('</MPD>')
    with open(mpd_path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {mpd_path}: {len(rungs)} reps, {nseg} segments, {dur}s", flush=True)


def complete(rootdir, r, expect):
    n, _ = rung_stats(rootdir, r)
    return expect is not None and n >= expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True, help="output root (a tos_2s_<tag>/ dir is created)")
    ap.add_argument("--workdir", required=True, help="where the 1080p mezzanine lives")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=0)
    ap.add_argument("--rungs", default=DEFAULT_RUNGS)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tag", default="1080p", help="dir/mpd suffix (kept '1080p' so the page's naming rule works)")
    ap.add_argument("--skip-mezz", action="store_true")
    ap.add_argument("--mpd-only", action="store_true")
    ap.add_argument("--expect-segments", type=int, default=None,
                    help="skip rungs that already have this many segments (resume)")
    a = ap.parse_args()

    rungs = parse_rungs(a.rungs)
    rootdir = os.path.join(a.out, f"{PREFIX}_2s_{a.tag}")
    os.makedirs(rootdir, exist_ok=True)
    os.makedirs(a.workdir, exist_ok=True)
    mezz = os.path.join(a.workdir, f"{PREFIX}_mezz_1080p.mp4")
    if not a.mpd_only:
        todo = [(r, h) for r, h in rungs if not complete(rootdir, r, a.expect_segments)]
        if any(h == 1080 for _, h in todo) and not (a.skip_mezz and os.path.exists(mezz)):
            make_mezz(a.src, mezz, a.start, a.dur)
        for h in sorted({h for _, h in todo}):
            rs = [r for r, hh in todo if hh == h]
            bs = a.batch if h <= 1440 else (2 if h == 2160 else 1)
            for i in range(0, len(rs), bs):
                encode_batch(mezz if h == 1080 else a.src, a.start, a.dur, rootdir, rs[i:i + bs], h)
    write_mpd(rootdir, os.path.join(a.out, f"{PREFIX}_ibmff_mixed_{a.tag}.mpd"),
              f"http://127.0.0.1:8080/{PREFIX}_2s_{a.tag}/", rungs)
    write_mpd(rootdir, os.path.join(a.out, f"{PREFIX}_ibmff_mixed_{a.tag}-public-access.mpd"),
              f"{PREFIX}_2s_{a.tag}/", rungs)


if __name__ == "__main__":
    sys.exit(main())
