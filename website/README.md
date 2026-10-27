# JAM project website

Source for the [project page](https://y0uroy.github.io/JAM/): interactive method diagrams, benchmark tables, recorded rollouts, probing and attention analyses, and a 59-second silent trailer.

## Preview and edit

From the repository root:

```bash
python3 -m http.server 8767 --bind 127.0.0.1 --directory website
```

Open `http://localhost:8767/`. All fonts, figures and videos are served locally; no CDN or JavaScript package installation is required.

Edit the files in `src/`, then regenerate the page with Python 3:

```bash
python3 website/src/build.py
python3 website/src/build.py --check
```

| Path | Contents |
|---|---|
| `src/page.html`, `src/styles.css`, `src/app.js` | Layout, styling and interactions |
| `src/data.js` | Paper values shown on the page, each with its table or section |
| `src/media.json` | Relative asset paths, durations, dimensions and file sizes |
| `index.html` | Generated, self-contained page code |
| `assets/rollouts/`, `assets/analysis/`, `assets/trailer/` | MP4/WebM video and posters |
| `assets/fonts/` | Self-hosted fonts, source URLs and OFL licenses |
| `assets/figures/fig2_method.png` | Method figure used in the repository README |

For static hosting, publish `index.html` and `assets/`. GitHub Actions publishes only these files to https://y0uroy.github.io/JAM/ through `.github/workflows/pages.yml`.

The video loader fetches the trailer as a Blob before playback so seeking works on hosts without HTTP Range support. LIBERO rollouts keep one frame per 20 Hz control step; the RoboTwin pairs keep one frame per control step and play at 10 control steps per second. The paper has been submitted to arXiv and is processing; its public identifier is pending.

On smaller screens, the contents menu links directly to all ten sections. Probe and attention players can expand without reloading their video; background playback pauses while a viewer is open or the tab is hidden. Save-Data and reduced-motion preferences suppress autoplay. Full benchmark and paired-category tables export their currently displayed values as CSV. The Code and Model buttons link to the GitHub and Hugging Face repositories.

Code follows the repository license and upstream notices. Fonts retain the included OFL licenses. Research media remain attributed to their authors; benchmark assets retain their respective terms. Keep model weights and datasets outside this folder.
