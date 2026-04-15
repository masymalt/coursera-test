# coursera-test
coursera test repository

## Lyrics Structure Pipeline

`lyrics_structure_pipeline.py` builds song section markers (`verse/chorus/...`) by combining:

- audio structure analysis (`librosa` + clustering),
- internet lyrics lookup (`lrclib`, fallback `lyrics.ovh`),
- optional local lyrics files (`.lrc`/`.txt`).

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U numpy scipy pandas scikit-learn librosa soundfile matplotlib requests
```

### Run

```bash
python lyrics_structure_pipeline.py "/path/to/song.mp3" \
  --artist "Artist" \
  --title "Track Title" \
  --out-dir "./analysis_out"
```

If `--artist/--title` are omitted, the script tries to parse filename format:

`Artist - Title.mp3`

### Optional local lyrics

```bash
python lyrics_structure_pipeline.py "/path/to/song.mp3" \
  --lyrics-file "/path/to/lyrics.lrc" \
  --out-dir "./analysis_out"
```

### Output files

- `structure.png` – waveform with colorized sections
- `structure_player.html` – interactive waveform player with clickable section blocks
- `markers.csv` – section markers with source tags
- `markers.cue` – CUE markers
- `summary.txt` – short time-based summary
- `lyrics_metadata.json` – provider/debug metadata

Open `structure_player.html` in a browser to play audio and jump by section.
