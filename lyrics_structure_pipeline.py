#!/usr/bin/env python3
"""Audio structure + internet lyrics pipeline.

Pipeline:
1) Segment the song into structural blocks from audio.
2) Fetch lyrics from the internet (LRCLIB first, lyrics.ovh fallback).
3) If timed lyrics with section tags are available, use them for labeling.
4) If only plain lyrics with [Verse]/[Chorus] tags are available, align by sequence.
5) Export markers (CSV + CUE) and a structure plot.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SECTION_ALIASES = {
    "verse": "verse",
    "куплет": "verse",
    "chorus": "chorus",
    "hook": "chorus",
    "refrain": "chorus",
    "припев": "chorus",
    "pre-chorus": "pre_chorus",
    "pre chorus": "pre_chorus",
    "bridge": "bridge",
    "бридж": "bridge",
    "intro": "intro",
    "outro": "outro",
}


@dataclass
class TimedLyricLine:
    time_sec: float
    text: str


@dataclass
class SectionInterval:
    start: float
    end: float
    label: str


@dataclass
class AudioSegment:
    start: float
    end: float
    duration: float
    vec: np.ndarray
    energy: float
    cluster: int = -1
    pattern: str = ""
    label: str = "section"
    label_source: str = "audio"


def choose_num_segments(duration_sec: float) -> int:
    k = int(round(duration_sec / 20.0))
    return int(np.clip(k, 5, 14))


def sec_to_cue(seconds: float) -> str:
    total_frames = int(round(seconds * 75))
    mm = total_frames // (60 * 75)
    ss = (total_frames // 75) % 60
    ff = total_frames % 75
    return f"{mm:02d}:{ss:02d}:{ff:02d}"


def sec_to_mmss_mmm(seconds: float) -> str:
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ss += 1
        ms = 0
    if ss >= 60:
        mm += 1
        ss -= 60
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def normalize_rows(x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(x.T).T


def parse_filename_metadata(audio_path: Path) -> tuple[str | None, str | None]:
    stem = audio_path.stem.strip()
    if " - " not in stem:
        return None, None
    artist, title = stem.split(" - ", 1)
    artist = artist.strip() or None
    title = title.strip() or None
    return artist, title


def canonical_section_label(raw: str) -> str | None:
    token = raw.strip().lower()
    token = token.replace("_", " ").replace("—", "-")
    token = re.sub(r"\s+", " ", token)
    for key, value in SECTION_ALIASES.items():
        if token.startswith(key):
            return value
    return None


def extract_section_label_from_line(text: str) -> str | None:
    # Examples: [Verse], (Chorus), Verse 2:
    bracket_match = re.search(r"[\[\(]\s*([a-zA-Zа-яА-Я\- ]{3,20})\s*[\]\)]", text)
    if bracket_match:
        label = canonical_section_label(bracket_match.group(1))
        if label:
            return label

    colon_match = re.match(r"^\s*([a-zA-Zа-яА-Я\- ]{3,20})\s*:\s*$", text)
    if colon_match:
        label = canonical_section_label(colon_match.group(1))
        if label:
            return label

    return None


def parse_lrc_timed_lines(text: str) -> list[TimedLyricLine]:
    lines: list[TimedLyricLine] = []
    timestamp_re = re.compile(r"\[(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\]")
    for raw_line in text.splitlines():
        matches = list(timestamp_re.finditer(raw_line))
        if not matches:
            continue
        lyric_text = timestamp_re.sub("", raw_line).strip()
        for match in matches:
            mm = int(match.group(1))
            ss = float(match.group(2))
            time_sec = mm * 60 + ss
            lines.append(TimedLyricLine(time_sec=time_sec, text=lyric_text))
    lines.sort(key=lambda x: x.time_sec)
    return lines


def normalize_lyric_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z0-9а-яё\s]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_intervals_from_timed_lyrics(
    timed_lines: list[TimedLyricLine], duration_sec: float
) -> list[SectionInterval]:
    starts: list[tuple[float, str]] = []
    for line in timed_lines:
        label = extract_section_label_from_line(line.text)
        if label:
            starts.append((line.time_sec, label))
    if not starts:
        return []

    intervals: list[SectionInterval] = []
    for idx, (start, label) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else duration_sec
        if end > start:
            intervals.append(SectionInterval(start=start, end=end, label=label))
    return intervals


def infer_chorus_intervals_from_repetition(
    timed_lines: list[TimedLyricLine], duration_sec: float
) -> list[SectionInterval]:
    """Infer chorus-like intervals from repeated synchronized lyric lines.

    Useful when LRC has timings but no explicit section tags.
    """
    normalized: list[tuple[float, str]] = []
    for line in timed_lines:
        token = normalize_lyric_text(line.text)
        if token and len(token) >= 8:
            normalized.append((line.time_sec, token))
    if not normalized:
        return []

    counts: dict[str, int] = {}
    for _, token in normalized:
        counts[token] = counts.get(token, 0) + 1

    repeated_anchor_times = sorted(
        [time_sec for time_sec, token in normalized if counts.get(token, 0) >= 2]
    )
    if len(repeated_anchor_times) < 2:
        return []

    groups: list[list[float]] = [[repeated_anchor_times[0]]]
    for time_sec in repeated_anchor_times[1:]:
        if time_sec - groups[-1][-1] <= 14.0:
            groups[-1].append(time_sec)
        else:
            groups.append([time_sec])

    intervals: list[SectionInterval] = []
    for group in groups:
        if len(group) < 2:
            continue
        start = max(0.0, group[0] - 2.0)
        end = min(duration_sec, group[-1] + 10.0)
        if end - start >= 8.0:
            intervals.append(SectionInterval(start=start, end=end, label="chorus"))
    return intervals


def build_lyric_presence_intervals(
    timed_lines: list[TimedLyricLine], duration_sec: float
) -> list[SectionInterval]:
    non_empty = [line for line in timed_lines if normalize_lyric_text(line.text)]
    if not non_empty:
        return []

    intervals: list[SectionInterval] = []
    for idx, line in enumerate(non_empty):
        next_time = non_empty[idx + 1].time_sec if idx + 1 < len(non_empty) else duration_sec
        gap = max(1.0, min(8.0, next_time - line.time_sec))
        end = min(duration_sec, line.time_sec + gap)
        if end > line.time_sec:
            intervals.append(SectionInterval(start=line.time_sec, end=end, label="voice"))
    return intervals


def extract_section_sequence_from_plain_lyrics(text: str) -> list[str]:
    seq: list[str] = []
    for line in text.splitlines():
        label = extract_section_label_from_line(line.strip())
        if label and (not seq or seq[-1] != label):
            seq.append(label)
    return seq


def fetch_lrclib(
    artist: str, title: str, duration_sec: float | None = None
) -> dict[str, Any] | None:
    base_url = "https://lrclib.net/api"
    with requests.Session() as session:
        try:
            params = {"artist_name": artist, "track_name": title}
            if duration_sec is not None:
                params["duration"] = str(int(round(duration_sec)))
            response = session.get(f"{base_url}/get", params=params, timeout=12)
            if response.ok:
                payload = response.json()
                if isinstance(payload, dict) and (
                    payload.get("syncedLyrics") or payload.get("plainLyrics")
                ):
                    payload["_provider"] = "lrclib"
                    return payload
        except requests.RequestException:
            pass

        try:
            response = session.get(
                f"{base_url}/search",
                params={"artist_name": artist, "track_name": title},
                timeout=12,
            )
            if not response.ok:
                return None
            items = response.json()
            if not isinstance(items, list) or not items:
                return None

            def score(item: dict[str, Any]) -> tuple[float, int]:
                name_bonus = 0.0
                candidate_title = str(item.get("trackName", "")).lower()
                candidate_artist = str(item.get("artistName", "")).lower()
                if title.lower() in candidate_title:
                    name_bonus += 2.0
                if artist.lower() in candidate_artist:
                    name_bonus += 2.0
                duration_penalty = 0.0
                if duration_sec is not None and item.get("duration"):
                    duration_penalty = abs(float(item["duration"]) - duration_sec) / 30.0
                return (name_bonus - duration_penalty, 1 if item.get("syncedLyrics") else 0)

            best = sorted(items, key=score, reverse=True)[0]
            if best.get("syncedLyrics") or best.get("plainLyrics"):
                best["_provider"] = "lrclib"
                return best
        except requests.RequestException:
            return None
    return None


def fetch_lyrics_ovh(artist: str, title: str) -> dict[str, Any] | None:
    try:
        url = f"https://api.lyrics.ovh/v1/{requests.utils.quote(artist)}/{requests.utils.quote(title)}"
        response = requests.get(url, timeout=12)
        if not response.ok:
            return None
        payload = response.json()
        lyrics = payload.get("lyrics")
        if lyrics and isinstance(lyrics, str):
            return {"plainLyrics": lyrics, "_provider": "lyrics.ovh"}
    except requests.RequestException:
        return None
    return None


def cluster_segments(segment_vectors: np.ndarray) -> np.ndarray:
    n = len(segment_vectors)
    if n <= 2:
        return np.arange(n)
    max_k = min(6, n - 1)
    best_labels: np.ndarray | None = None
    best_score = -1.0
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(segment_vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(segment_vectors, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    return best_labels if best_labels is not None else np.arange(n)


def detect_audio_segments(audio_path: Path) -> tuple[np.ndarray, int, float, list[AudioSegment]]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    hop = 512

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)

    feats = np.vstack([chroma, mfcc, np.log1p(rms), np.log1p(centroid)])
    feats = normalize_rows(feats)

    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 8:
        step = max(1, feats.shape[1] // 200)
        beats = np.arange(0, feats.shape[1], step, dtype=int)
    if beats[0] != 0:
        beats = np.r_[0, beats]
    if beats[-1] != feats.shape[1] - 1:
        beats = np.r_[beats, feats.shape[1] - 1]

    feats_sync = librosa.util.sync(feats, beats, aggregate=np.median)
    rms_sync = librosa.util.sync(rms, beats, aggregate=np.mean).squeeze()
    k_segments = choose_num_segments(duration_sec)
    boundaries = librosa.segment.agglomerative(feats_sync, k=k_segments)
    boundaries = np.unique(np.asarray(boundaries, dtype=int))
    boundaries = boundaries[(boundaries >= 0) & (boundaries < feats_sync.shape[1])]
    if len(boundaries) == 0 or boundaries[0] != 0:
        boundaries = np.r_[0, boundaries]
    if boundaries[-1] != feats_sync.shape[1] - 1:
        boundaries = np.r_[boundaries, feats_sync.shape[1] - 1]

    segments: list[AudioSegment] = []
    for idx in range(len(boundaries) - 1):
        b_start = int(boundaries[idx])
        b_end = int(max(boundaries[idx + 1], b_start + 1))
        frame_start = int(beats[b_start])
        frame_end = int(beats[b_end])
        start = float(librosa.frames_to_time(frame_start, sr=sr, hop_length=hop))
        end = float(librosa.frames_to_time(frame_end, sr=sr, hop_length=hop))
        if end <= start:
            continue
        vec = feats_sync[:, b_start:b_end].mean(axis=1)
        energy = float(np.mean(rms_sync[b_start:b_end])) if b_end > b_start else float(rms_sync[b_start])
        segments.append(
            AudioSegment(
                start=start,
                end=end,
                duration=end - start,
                vec=vec,
                energy=energy,
            )
        )

    if len(segments) < 2:
        raise RuntimeError("Too few segments detected from audio.")

    segment_vectors = np.vstack([s.vec for s in segments])
    cluster_ids = cluster_segments(segment_vectors)
    seen: list[int] = []
    for cid in cluster_ids:
        if int(cid) not in seen:
            seen.append(int(cid))
    cid_to_pattern = {
        cid: (chr(ord("A") + i) if i < 26 else f"S{i+1}")
        for i, cid in enumerate(seen)
    }
    for segment, cid in zip(segments, cluster_ids):
        segment.cluster = int(cid)
        segment.pattern = cid_to_pattern[int(cid)]

    return y, sr, duration_sec, segments


def apply_audio_only_labels(segments: list[AudioSegment]) -> None:
    stats: dict[int, dict[str, Any]] = {}
    for seg in segments:
        st = stats.setdefault(seg.cluster, {"occ": 0, "dur": 0.0, "energies": [], "positions": []})
        st["occ"] += 1
        st["dur"] += seg.duration
        st["energies"].append(seg.energy)
        st["positions"].append(seg.start)

    for st in stats.values():
        st["mean_energy"] = float(np.mean(st["energies"]))

    repeated = [cid for cid, st in stats.items() if st["occ"] >= 2]
    chorus_cluster: int | None = None
    verse_cluster: int | None = None

    if repeated:
        occ = np.array([stats[c]["occ"] for c in repeated], dtype=float)
        ene = np.array([stats[c]["mean_energy"] for c in repeated], dtype=float)
        dur = np.array([stats[c]["dur"] for c in repeated], dtype=float)

        def z(values: np.ndarray) -> np.ndarray:
            std = float(np.std(values))
            return (values - np.mean(values)) / (std + 1e-9)

        score = 1.8 * z(occ) + 1.0 * z(ene) + 0.7 * z(dur)
        chorus_cluster = repeated[int(np.argmax(score))]
        verse_candidates = [c for c in repeated if c != chorus_cluster]
        if verse_candidates:
            verse_cluster = min(verse_candidates, key=lambda c: stats[c]["mean_energy"])

    last_idx = len(segments) - 1
    for idx, seg in enumerate(segments):
        if idx == 0 and seg.duration < 24 and stats[seg.cluster]["occ"] == 1:
            seg.label = "intro"
        elif idx == last_idx and seg.duration < 30 and stats[seg.cluster]["occ"] == 1:
            seg.label = "outro"
        elif chorus_cluster is not None and seg.cluster == chorus_cluster:
            seg.label = "chorus"
        elif verse_cluster is not None and seg.cluster == verse_cluster:
            seg.label = "verse"
        else:
            seg.label = f"section_{seg.pattern.lower()}"
        seg.label_source = "audio"


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def apply_timed_lyrics_labels(
    segments: list[AudioSegment], intervals: list[SectionInterval], min_overlap_ratio: float = 0.2
) -> bool:
    if not intervals:
        return False
    changed = False
    for seg in segments:
        overlaps: list[tuple[float, str]] = []
        for interval in intervals:
            ov = overlap_seconds(seg.start, seg.end, interval.start, interval.end)
            if ov > 0:
                overlaps.append((ov, interval.label))
        if not overlaps:
            continue
        best_overlap, best_label = max(overlaps, key=lambda x: x[0])
        if best_overlap / max(seg.duration, 1e-6) >= min_overlap_ratio:
            seg.label = best_label
            seg.label_source = "lyrics_timed"
            changed = True
    return changed


def apply_verse_labels_from_lyric_presence(
    segments: list[AudioSegment], lyric_presence: list[SectionInterval]
) -> bool:
    if not lyric_presence:
        return False

    changed = False
    for seg in segments:
        if seg.label in {"chorus", "intro", "outro"}:
            continue
        overlap = sum(
            overlap_seconds(seg.start, seg.end, interval.start, interval.end)
            for interval in lyric_presence
        )
        if overlap / max(seg.duration, 1e-6) >= 0.35:
            seg.label = "verse"
            if seg.label_source == "audio":
                seg.label_source = "lyrics_inferred"
            changed = True
    return changed


def apply_plain_lyrics_sequence_labels(segments: list[AudioSegment], section_seq: list[str]) -> bool:
    if not section_seq or len(segments) < 2:
        return False

    n = len(segments)
    m = len(section_seq)
    if m == 1:
        dominant = section_seq[0]
        for seg in segments:
            seg.label = dominant
            seg.label_source = "lyrics_sequence"
        return True

    votes: dict[int, dict[str, int]] = {}
    for i, section_label in enumerate(section_seq):
        mapped_index = int(round(i * (n - 1) / (m - 1)))
        cluster = segments[mapped_index].cluster
        cluster_votes = votes.setdefault(cluster, {})
        cluster_votes[section_label] = cluster_votes.get(section_label, 0) + 1

    cluster_to_label: dict[int, str] = {}
    for cluster, cluster_votes in votes.items():
        cluster_to_label[cluster] = max(cluster_votes.items(), key=lambda item: item[1])[0]

    changed = False
    for seg in segments:
        mapped = cluster_to_label.get(seg.cluster)
        if mapped:
            seg.label = mapped
            seg.label_source = "lyrics_sequence"
            changed = True
    return changed


def smooth_short_islands(segments: list[AudioSegment], max_short_duration: float = 12.0) -> None:
    labels = [s.label for s in segments]
    for i in range(1, len(segments) - 1):
        prev_label = labels[i - 1]
        next_label = labels[i + 1]
        if prev_label == next_label and labels[i] != prev_label and segments[i].duration <= max_short_duration:
            labels[i] = prev_label
    for seg, label in zip(segments, labels):
        seg.label = label


def merge_adjacent_same_labels(segments: list[AudioSegment]) -> list[AudioSegment]:
    if not segments:
        return []
    merged: list[AudioSegment] = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        if current.label == prev.label:
            prev.end = current.end
            prev.duration = prev.end - prev.start
            if prev.label_source != current.label_source:
                prev.label_source = "mixed"
        else:
            merged.append(current)
    return merged


def plot_structure(y: np.ndarray, sr: int, segments: list[AudioSegment], output_png: Path) -> None:
    duration_sec = len(y) / sr
    times = np.linspace(0, duration_sec, num=len(y))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_range = max(1e-6, y_max - y_min)

    plt.figure(figsize=(16, 4.5))
    plt.plot(times, y, color="black", linewidth=0.55, alpha=0.8)

    labels_order = list(dict.fromkeys(seg.label for seg in segments))
    cmap = plt.get_cmap("tab20")
    color_map = {label: cmap(i % 20) for i, label in enumerate(labels_order)}

    for seg in segments:
        color = color_map[seg.label]
        plt.axvspan(seg.start, seg.end, color=color, alpha=0.3, linewidth=0)
        mid = (seg.start + seg.end) / 2.0
        plt.text(
            mid,
            y_min + 0.15 * y_range,
            seg.label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    plt.title("Song structure (audio + lyrics)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.xlim(0, duration_sec)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def write_outputs(
    audio_path: Path,
    out_dir: Path,
    segments: list[AudioSegment],
    metadata: dict[str, Any],
) -> None:
    rows = []
    for seg in segments:
        rows.append(
            {
                "start_sec": round(seg.start, 3),
                "end_sec": round(seg.end, 3),
                "duration_sec": round(seg.duration, 3),
                "label": seg.label,
                "pattern": seg.pattern,
                "cluster_id": seg.cluster,
                "label_source": seg.label_source,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "markers.csv", index=False)

    cue_lines = [
        'PERFORMER "Unknown"',
        f'TITLE "{audio_path.stem}"',
        f'FILE "{audio_path.name}" {audio_path.suffix.lstrip(".").upper() or "MP3"}',
    ]
    for idx, seg in enumerate(segments, start=1):
        cue_lines.append(f"  TRACK {idx:02d} AUDIO")
        cue_lines.append(f'    TITLE "{seg.label}"')
        cue_lines.append(f"    INDEX 01 {sec_to_cue(seg.start)}")
    (out_dir / "markers.cue").write_text("\n".join(cue_lines), encoding="utf-8")

    summary_lines = [f"Audio: {audio_path.name}", ""]
    for seg in segments:
        summary_lines.append(
            f"{sec_to_mmss_mmm(seg.start)} - {sec_to_mmss_mmm(seg.end)}  {seg.label:<12} [{seg.label_source}]"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    (out_dir / "lyrics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline(
    audio_path: Path,
    out_dir: Path,
    artist: str | None,
    title: str | None,
    lyrics_file: Path | None,
    disable_web_lyrics: bool,
) -> None:
    y, sr, duration_sec, segments = detect_audio_segments(audio_path)
    apply_audio_only_labels(segments)

    metadata: dict[str, Any] = {
        "artist": artist,
        "title": title,
        "lyrics_provider": None,
        "timed_lyrics_found": False,
        "section_tags_found": False,
    }

    synced_text: str | None = None
    plain_text: str | None = None

    if lyrics_file is not None:
        text = lyrics_file.read_text(encoding="utf-8", errors="replace")
        if lyrics_file.suffix.lower() == ".lrc":
            synced_text = text
        else:
            plain_text = text
        metadata["lyrics_provider"] = f"local:{lyrics_file.name}"
    elif not disable_web_lyrics and artist and title:
        payload = fetch_lrclib(artist, title, duration_sec=duration_sec)
        if payload is None:
            payload = fetch_lyrics_ovh(artist, title)
        if payload:
            synced_text = payload.get("syncedLyrics")
            plain_text = payload.get("plainLyrics") or payload.get("lyrics")
            metadata["lyrics_provider"] = payload.get("_provider")
            metadata["lyrics_response"] = {
                k: v
                for k, v in payload.items()
                if k in {"trackName", "artistName", "albumName", "duration", "_provider"}
            }

    used_lyrics = False
    if synced_text:
        timed_lines = parse_lrc_timed_lines(synced_text)
        timed_intervals = build_intervals_from_timed_lyrics(timed_lines, duration_sec)
        metadata["timed_lyrics_found"] = bool(timed_lines)
        metadata["section_tags_found"] = bool(timed_intervals)
        if timed_intervals:
            if apply_timed_lyrics_labels(segments, timed_intervals):
                used_lyrics = True
        elif timed_lines:
            chorus_intervals = infer_chorus_intervals_from_repetition(timed_lines, duration_sec)
            if chorus_intervals and apply_timed_lyrics_labels(segments, chorus_intervals):
                used_lyrics = True
                metadata["section_tags_found"] = True
                metadata["inferred_sections_from_repetition"] = True
            lyric_presence = build_lyric_presence_intervals(timed_lines, duration_sec)
            if apply_verse_labels_from_lyric_presence(segments, lyric_presence):
                used_lyrics = True

    if not used_lyrics and plain_text:
        section_seq = extract_section_sequence_from_plain_lyrics(plain_text)
        metadata["section_tags_found"] = bool(section_seq)
        if apply_plain_lyrics_sequence_labels(segments, section_seq):
            used_lyrics = True

    smooth_short_islands(segments)
    segments = merge_adjacent_same_labels(segments)

    if used_lyrics:
        metadata["labeling_mode"] = "audio+lyrics"
    else:
        metadata["labeling_mode"] = "audio-only"

    plot_structure(y, sr, segments, out_dir / "structure.png")
    write_outputs(audio_path, out_dir, segments, metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find song sections and label them using audio + internet lyrics."
    )
    parser.add_argument("audio", type=str, help="Path to audio file (mp3/wav/flac)")
    parser.add_argument("--out-dir", type=str, default="analysis_out", help="Output directory")
    parser.add_argument("--artist", type=str, default=None, help="Artist name (for lyrics search)")
    parser.add_argument("--title", type=str, default=None, help="Track title (for lyrics search)")
    parser.add_argument(
        "--lyrics-file",
        type=str,
        default=None,
        help="Optional local .lrc/.txt file with lyrics and section tags",
    )
    parser.add_argument(
        "--disable-web-lyrics",
        action="store_true",
        help="Disable internet lyrics fetching and use audio only (or --lyrics-file)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artist = args.artist
    title = args.title
    if not artist or not title:
        inferred_artist, inferred_title = parse_filename_metadata(audio_path)
        artist = artist or inferred_artist
        title = title or inferred_title

    lyrics_file = Path(args.lyrics_file).expanduser().resolve() if args.lyrics_file else None
    if lyrics_file and not lyrics_file.exists():
        raise FileNotFoundError(f"Lyrics file not found: {lyrics_file}")

    run_pipeline(
        audio_path=audio_path,
        out_dir=out_dir,
        artist=artist,
        title=title,
        lyrics_file=lyrics_file,
        disable_web_lyrics=args.disable_web_lyrics,
    )
    print("Done.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
