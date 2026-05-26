# Deployment guide

This project is designed to deploy to two free-tier services without code
changes. Pick whichever is easier; both link back to the same `app.py`.

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to <https://streamlit.io/cloud> → **New app**.
3. Repository: `your-handle/text2qca` · Branch: `main` · Main file: `app.py`.
4. **Deploy**. Streamlit Cloud auto-installs `requirements.txt`. First build
   takes ~3 minutes; subsequent restarts are instant.
5. Copy the public URL into the README's **Live demo** section.

## HuggingFace Spaces

1. Visit <https://huggingface.co/new-space>.
2. **Space SDK** = Streamlit. **Space hardware** = CPU basic (free) is enough.
3. Create the Space, then clone it locally:
   ```bash
   git clone https://huggingface.co/spaces/your-handle/text2qca
   ```
4. Copy this repository's contents into the cloned folder (the README's YAML
   frontmatter is already in HF Spaces format).
5. ```bash
   git add . && git commit -m "Initial deploy" && git push
   ```
6. The Space rebuilds itself on every push. First build downloads the
   `BAAI/bge-small-zh-v1.5` model (~95 MB) into HF's persistent cache, so
   subsequent runs are fast.

## Notes on resource usage

- **Memory** — `torch` + `bge-small-zh-v1.5` need about 700 MB RAM. Both
  Streamlit Cloud and HF Spaces free tiers (1 GB) handle this; if you hit
  OOM, switch the scoring method to `keyword` in the sidebar.
- **Cold-start** — first user request after the Space sleeps re-downloads
  the model. Subsequent requests are cached.
- **Auth / secrets** — none required. The tool is fully self-contained.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OSError: Can't load model` | Set scoring method to `keyword` in the sidebar — the tool falls back automatically, but explicitly choosing it avoids the long timeout. |
| Excel export raises `openpyxl` error | Ensure `openpyxl>=3.1` is installed; included in `requirements.txt`. |
| Chinese characters render as squares in figures | Install a CJK font (`apt-get install fonts-noto-cjk` on Linux); the figure helper falls back to `DejaVu Sans` when no CJK font is found. |
