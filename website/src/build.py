#!/usr/bin/env python3
"""Bundle the editable JAM page with Python 3's standard library.

From the repository root: python3 website/src/build.py
For a separate hosting checkout: python3 src/build.py --site-dir dist
Use --check to verify the committed page without changing it.
"""
import argparse
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
ORIGIN = "https://y0uroy.github.io/JAM"


def build(site):
    media = json.loads((SRC / "media.json").read_text(encoding="utf-8"))
    trailer, probe, att = (media[key] for key in ("trailer", "probe", "att"))
    seconds = round(trailer["dur"])
    duration = f"{seconds // 60}:{seconds % 60:02}"
    poster = trailer["poster"]

    def aspect(item, margin):
        w, h = item["w"], item["h"]
        cap = f"; max-width: calc((100vh - {margin}px) * {w} / {h})" if h > w else ""
        return f"{w} / {h}{cap}"

    replacements = {
        "%%TRAILER_POSTER_ATTR%%": f'poster="{poster}"',
        "%%TRAILER_POSTER_IMG%%": f'<img class="film-poster" src="{poster}" alt="">',
        "%%TRAILER_PH_HIDDEN%%": "hidden",
        "%%TRAILER_BTN_HIDDEN%%": "",
        "%%TRAILER_CAPTION%%": f"Trailer · {duration} · Silent",
        "%%TRAILER_LEN%%": duration,
        "%%PROBE_AR%%": aspect(probe, 150),
        "%%PROBE_POSTER%%": probe["poster"],
        "%%ATT_AR%%": aspect(att, 280),
        "%%ATT_POSTER%%": att["poster"],
    }
    body = (SRC / "page.html").read_text(encoding="utf-8")
    for marker, value in replacements.items():
        if marker not in body:
            raise ValueError(f"Missing template marker: {marker}")
        body = body.replace(marker, value)
    if "%%" in body:
        raise ValueError("Unresolved template marker")
    for item in [trailer, probe, att]:
        for ext, size in item["bytes"].items():
            path = site / f'{item["base"]}.{ext}'
            if path.stat().st_size != size:
                raise ValueError(f"Media manifest mismatch: {path}")
    for slug, pair in media["rollouts"].items():
        for side, formats in pair.items():
            for ext, size in formats.items():
                path = site / f"assets/rollouts/{slug}-{side}.{ext}"
                if path.stat().st_size != size:
                    raise ValueError(f"Rollout manifest mismatch: {path}")

    css = (SRC / "styles.css").read_text(encoding="utf-8").strip()
    data = (SRC / "data.js").read_text(encoding="utf-8").strip()
    app = (SRC / "app.js").read_text(encoding="utf-8").strip()
    js = data + "\nconst M = " + json.dumps(media, ensure_ascii=False, separators=(",", ":")) + ";\n" + app
    if "</script" in js.lower() or "</style" in css.lower():
        raise ValueError("Unexpected closing tag in source")
    description = "Learn from the future. Never generate it in training. JAM trains action policies with foresight supervision from recorded observations."
    head = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#080b12">
<title>JAM · Just Action Models</title>
<meta name="description" content="{description}">
<link rel="canonical" href="{ORIGIN}/">
<meta property="og:type" content="website">
<meta property="og:title" content="JAM · Just Action Models">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{ORIGIN}/">
<meta property="og:image" content="{ORIGIN}/{poster}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/fonts/fonts.css">
<style>
:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}
body{{margin:0;font:14px system-ui}}img{{max-width:100%}}[hidden]{{display:none!important}}
{css}
</style>
</head>
<body>'''
    return head + "\n" + body.strip() + "\n<script>\n" + js + "\n</script>\n</body>\n</html>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=SRC.parent)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.site_dir)
    target = args.site_dir / "index.html"
    if args.check:
        if target.read_text(encoding="utf-8") != result:
            raise SystemExit("index.html is stale; run build.py first")
        print("Page and media manifest are current.")
    else:
        target.write_text(result, encoding="utf-8")
        print(f"Built {target} ({len(result.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
