"""Referee-whistle detector on the tactical video's audio track.

Findings that shaped it (all from the match audio itself, no labels):
  * the tactical audio is very quiet (peaks ~ -44 dBFS) but tonal sounds stand out clearly in a
    spectrogram regardless of loudness;
  * the stadium is full of *crowd* whistling (2.7-3.4 kHz, pitch wanders, comes in bursts);
  * among stable-pitch tonal events with a harmonic at 2x the fundamental, the whole-match
    histogram has a separate cluster at ~3.7-3.9 kHz, split from the crowd mass by a dip at
    3.5-3.7 kHz -- the referee's whistle (one referee, one whistle, one pitch).

Score per audio frame (10 ms) = min(prominence of the fundamental, prominence of its 2nd
harmonic), on a 100 ms-averaged spectrogram; prominence = level minus the median of +/-12 bins.
An event is >= 0.08 s above SCORE_ON inside REFEREE_BAND with a stable pitch. Settings were chosen
by recall on moments that must carry a whistle and are known without labels (the two published
kick-offs, the two half ends, the Smart-Tagging kick-offs after goals): 5 of 8 found, 86 events.

    python -m possession.audio          # writes data/cache/whistle.npz and prints checks
"""

import os
import subprocess

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft

from .data import CACHE_DIR
from games import GAMES, asset_paths, half_video_blocks

SR = 16000
NPERSEG = 1024
HOP = 160                     # 100 audio frames per second
SEARCH = (2700, 3990)         # where fundamentals are searched (2x must stay below 8 kHz)
REFEREE_BAND = (3650, 3990)   # cluster found in the whole-match pitch histogram
SCORE_ON = 8.0                # dB
MIN_LEN = 8                   # audio frames = 0.08 s
MAX_PITCH_STD = 60.0          # Hz
CHUNK_S = 120


def _read_audio(start_s, dur_s):
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(start_s), "-t", str(dur_s),
         "-i", asset_paths()["video"], "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32)


def tonal_track(a):
    """(time s, fundamental Hz, score dB) per 10 ms audio frame of a mono signal."""
    f, t, Z = stft(a, SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    db = 10 * np.log10(uniform_filter1d(np.abs(Z) ** 2, 10, axis=1) + 1e-12)
    band = np.flatnonzero((f >= SEARCH[0]) & (f <= SEARCH[1]))
    k = band[db[band].argmax(0)]
    cols = np.arange(db.shape[1])

    def prom(kk):
        nb = np.clip(kk[:, None] + np.arange(-12, 13), 0, len(f) - 1)
        return db[kk, cols] - np.median(db[nb, cols[:, None]], 1)

    score = np.minimum(prom(k), prom(np.clip(2 * k, 0, len(f) - 1)))
    return t, f[k], score


def detect(game="BAR-ATH", use_cache=True):
    """(score per video frame [referee band only, else 0], [(start_frame, end_frame, pitch)])."""
    path = os.path.join(CACHE_DIR, "whistle.npz")
    n = GAMES[game]["n_video_frames"]
    if use_cache and os.path.exists(path):
        z = np.load(path)
        return z["score"], [tuple(e) for e in z["events"]]
    score = np.zeros(n, np.float32)
    events = []
    for s0 in range(0, int(np.ceil(n / 25)), CHUNK_S):
        a = _read_audio(s0, CHUNK_S)
        if not len(a):
            break
        t, pitch, s = tonal_track(a)
        frames = np.minimum(((s0 + t) * 25).astype(int), n - 1)
        inband = (pitch >= REFEREE_BAND[0]) & (pitch <= REFEREE_BAND[1])
        np.maximum.at(score, frames, np.where(inband, np.maximum(s, 0), 0).astype(np.float32))
        on = (s > SCORE_ON) & (pitch >= REFEREE_BAND[0]) & (pitch <= REFEREE_BAND[1])
        e = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
        for a_, b_ in zip(e[::2], e[1::2]):
            p = pitch[a_:b_]
            if b_ - a_ >= MIN_LEN and np.std(p) < MAX_PITCH_STD and \
                    REFEREE_BAND[0] <= np.median(p) <= REFEREE_BAND[1]:
                events.append((int(frames[a_]), int(frames[b_ - 1]), float(np.median(p))))
    merged = []
    for ev in events:                              # one blast can be split by a short gap
        if merged and ev[0] - merged[-1][1] < 50:     # within 2 s = same signal
            merged[-1] = (merged[-1][0], ev[1], merged[-1][2])
        else:
            merged.append(ev)
    np.savez(path, score=score, events=np.array(merged, float).reshape(-1, 3))
    return score, merged


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "starter"))
    import load_smart_tagging as st

    score, events = detect(use_cache=False)
    starts = np.array([e[0] for e in events])
    print("referee whistle events:", len(events))
    blocks = half_video_blocks("BAR-ATH")
    for half, a, b in blocks:
        print("  half {}: {} inside play".format(half, int(np.sum((starts >= a) & (starts <= b)))))

    def nearest(frame):
        if not len(starts):
            return "none"
        d = starts - frame
        i = np.argmin(np.abs(d))
        return "{:+.1f}s".format(d[i] / 25)

    print("\nchecks against moments that must have a whistle (nearest detection):")
    for name, fr in (("kick-off 1st half (published)", 11185), ("kick-off 2nd half (published)", 113742),
                     ("end of 1st half", blocks[0][2]), ("end of 2nd half", blocks[1][2])):
        print("  {:<32} {}".format(name, nearest(fr)))
    for e in st.load():
        if (e["code"] or "").upper() == "SET PIECES" and e["tags"].get("Type", "").startswith(("Kick Off (After", "Free")):
            print("  {:<32} {}".format("tag " + e["tags"]["Type"], nearest(e["start_frame"] + 88)))
