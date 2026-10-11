"""
check_onset_and_topography_lab.py

Same visual QC checks (raw traces, origin-channel zoom, topomap sequence),
but now using the real pipeline loaders -- get_seizures_list() and
seizure_num_to_raw_data() from data_processing.py -- since you're on the
lab computer and DATA_PATH is reachable. That means you don't need to
manually find which .data file corresponds to a given seizure_num; the
existing pipeline logic resolves that for you via file_seizure_ind.

Two known bugs are worked around here rather than fixed in data_processing.py
itself, so this script is safe to run without editing that file:

  1. data_processing.py calls `mpl.use('QtAgg')` at import time, which
     crashes with ImportError if PyQt/PySide isn't installed. We patch
     matplotlib.use() to a no-op immediately before importing
     data_processing, so that call becomes harmless and we keep our own
     'Agg' backend instead.

  2. copy_and_crop() parses onset/offset with
     format='%Y-%m-%d %H:%M:%S', which does not match the real seizure
     tables ('DD/MM/YYYY HH:MM', dayfirst, minute precision, no seconds).
     We don't call copy_and_crop() at all here -- onset/offset are parsed
     directly with dayfirst=True instead.

IMPORTANT: data_processing.py's DATA_PATH constant is hardcoded to
    E:/Ben Gurion University Of Negev Dropbox/CPL lab members/epilepsy_data/Epilepsiea
If that's not the correct path for however this session reaches the lab
drive, edit that DATA_PATH line directly inside data_processing.py before
running this script -- seizure_num_to_raw_data() uses it internally and
doesn't take a path override.

Usage:
    Make sure this file sits in the same folder as data_processing.py.
    Edit PAT_NUM / SURF / SEIZURE_NUM at the bottom, then run:

        python check_onset_and_topography_lab.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # set our backend first...

# ...then patch matplotlib.use() to a no-op so data_processing.py's own
# `mpl.use('QtAgg')` call (which runs at import time) can't override it
# or crash if Qt bindings aren't installed.
_original_use = matplotlib.use
matplotlib.use = lambda *args, **kwargs: None

from data_processing import (  # noqa: E402  (import after backend patch, intentional)
    get_seizures_list,
    seizure_num_to_raw_data,
    fix_electrode_names,
)

matplotlib.use = _original_use  # restore, harmless either way at this point

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "onset_topography_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTAGE_CASE_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 'F3': 'F3', 'F4': 'F4', 'C3': 'C3', 'C4': 'C4',
    'P3': 'P3', 'P4': 'P4', 'O1': 'O1', 'O2': 'O2', 'F7': 'F7', 'F8': 'F8',
    'T7': 'T7', 'T8': 'T8', 'P7': 'P7', 'P8': 'P8', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
}


def load_and_prep(pat_num, surf, seizure_num):
    """Fetch the correct raw recording for this seizure via the real
    pipeline lookup (get_seizures_list + seizure_num_to_raw_data), then
    apply light filtering only (no average reference) so raw morphology
    around onset stays visible."""
    seizures_list_table = get_seizures_list(pat_num, surf)
    raw = seizure_num_to_raw_data(pat_num, seizure_num, seizures_list_table, surf)
    raw = fix_electrode_names(raw.copy())
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

    seizure_info = seizures_list_table[
        seizures_list_table['seizure_num'] == seizure_num
    ].iloc[0]
    return raw, seizure_info


def get_onset_offset_seconds(raw, seizure_info):
    recording_start = np.datetime64(raw.info['meas_date']).astype('datetime64[us]')

    # dayfirst=True, no fixed format: matches 'DD/MM/YYYY HH:MM'
    # (minute precision, no seconds) as seen in the real seizure tables.
    onset = np.datetime64(
        pd.to_datetime(str(seizure_info['onset']).strip(), dayfirst=True)
    ).astype('datetime64[us]')
    offset = np.datetime64(
        pd.to_datetime(str(seizure_info['offset']).strip(), dayfirst=True)
    ).astype('datetime64[us]')

    onset_sec = float((onset - recording_start) / np.timedelta64(1, 's'))
    offset_sec = float((offset - recording_start) / np.timedelta64(1, 's'))
    return onset_sec, offset_sec


def parse_origin_channels(seizure_info):
    origin = seizure_info.get('origin', None)
    if origin is None or pd.isna(origin):
        return None
    return [ch.strip().upper() for ch in str(origin).split(',') if ch.strip()]


def plot_raw_traces(raw, onset_sec, offset_sec, tag, pad_before=15, pad_after=60):
    tmin = max(0.0, onset_sec - pad_before)
    tmax = min(raw.times[-1], onset_sec + pad_after)
    raw_seg = raw.copy().crop(tmin=tmin, tmax=tmax)

    data = raw_seg.get_data() * 1e6
    times = raw_seg.times + tmin
    ch_names = raw_seg.ch_names
    n_ch = len(ch_names)

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 0.6 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i, ch in enumerate(ch_names):
        axes[i].plot(times, data[i], linewidth=0.5, color='black')
        axes[i].set_ylabel(ch, rotation=0, labelpad=28, fontsize=8, va='center')
        axes[i].axvline(onset_sec, color='red', linestyle='--', linewidth=1,
                         label='onset' if i == 0 else None)
        if tmin <= offset_sec <= tmax:
            axes[i].axvline(offset_sec, color='blue', linestyle='--', linewidth=1,
                             label='offset' if i == 0 else None)
        axes[i].set_yticks([])
        for spine in ('top', 'right', 'left'):
            axes[i].spines[spine].set_visible(False)

    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s, from recording start)')
    fig.suptitle(f"{tag} — raw traces around onset")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"traces_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved trace plot: {out_path}")


def plot_origin_zoom(raw, onset_sec, offset_sec, origin_channels, tag,
                      pad_before=15, pad_after=25):
    available = [ch for ch in origin_channels if ch in raw.ch_names]
    if not available:
        print(f"None of the origin channels {origin_channels} found in raw data; skipping zoom plot.")
        return

    tmin = max(0.0, onset_sec - pad_before)
    tmax = min(raw.times[-1], onset_sec + pad_after)
    raw_seg = raw.copy().pick(available).crop(tmin=tmin, tmax=tmax)
    data = raw_seg.get_data() * 1e6
    times = raw_seg.times + tmin

    fig, axes = plt.subplots(len(available), 1, figsize=(14, 2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for i, ch in enumerate(available):
        axes[i].plot(times, data[i], linewidth=0.6, color='black')
        axes[i].axvline(onset_sec, color='red', linestyle='--', linewidth=1,
                         label='onset' if i == 0 else None)
        if tmin <= offset_sec <= tmax:
            axes[i].axvline(offset_sec, color='blue', linestyle='--', linewidth=1,
                             label='offset' if i == 0 else None)
        axes[i].set_ylabel(ch, rotation=0, labelpad=25, fontsize=9, va='center')

    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s, from recording start)')
    fig.suptitle(f"{tag} — zoom on labeled origin channels {available}")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"zoom_origin_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved origin zoom plot: {out_path}")


def plot_topomap_sequence(raw, onset_sec, tag,
                           offsets=(-5, 0, 5, 15, 30, 45), window=2.0, band=(1, 40)):
    raw_topo = raw.copy()
    rename_map = {ch: MONTAGE_CASE_MAP[ch] for ch in raw_topo.ch_names if ch in MONTAGE_CASE_MAP}
    raw_topo.rename_channels(rename_map)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw_topo.set_montage(montage, on_missing='ignore')

    fig, axes = plt.subplots(1, len(offsets), figsize=(3 * len(offsets), 3.2))

    all_powers = []
    for off in offsets:
        center = onset_sec + off
        tmin = max(0.0, center - window / 2)
        tmax = min(raw_topo.times[-1], center + window / 2)
        seg = raw_topo.copy().crop(tmin=tmin, tmax=tmax)

        n_fft = max(4, min(len(seg.times), int(seg.info['sfreq'] * window)))
        psd = seg.compute_psd(method='welch', fmin=band[0], fmax=band[1],
                               n_fft=n_fft, verbose=False)
        power = psd.get_data().mean(axis=1)
        all_powers.append(power)

    vmin, vmax = float(np.min(all_powers)), float(np.max(all_powers))

    for ax, off, power in zip(axes, offsets, all_powers):
        mne.viz.plot_topomap(power, raw_topo.info, axes=ax, show=False,
                              vlim=(vmin, vmax), cmap='RdBu_r')
        label = f"onset{'+' if off >= 0 else ''}{off}s"
        ax.set_title(label, fontsize=9)

    fig.suptitle(f"{tag} — power topography ({band[0]}-{band[1]} Hz)")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"topomap_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved topomap sequence: {out_path}")


def check_onset(pat_num, surf, seizure_num):
    tag = f"pat{pat_num}_{surf}_sz{seizure_num}"
    print(f"\n=== Checking {tag} ===")

    raw, seizure_info = load_and_prep(pat_num, surf, seizure_num)
    onset_sec, offset_sec = get_onset_offset_seconds(raw, seizure_info)
    print(f"classif.: {seizure_info.get('classif.', 'n/a')}")
    print(f"Onset at {onset_sec:.1f}s, offset at {offset_sec:.1f}s "
          f"(duration {offset_sec - onset_sec:.1f}s)")

    plot_raw_traces(raw, onset_sec, offset_sec, tag)

    origin_channels = parse_origin_channels(seizure_info)
    if origin_channels:
        print(f"Labeled origin channels: {origin_channels}")
        plot_origin_zoom(raw, onset_sec, offset_sec, origin_channels, tag)

    plot_topomap_sequence(raw, onset_sec, tag)


if __name__ == "__main__":
    # ---- EDIT THESE: patient 102's first SG (FBTCS) seizure ----
    PAT_NUM = "1"       # base_pat_num, matches folder name "pat_1" and "1_surf30_seizures.csv"
    SURF = "surf30"
    SEIZURE_NUM = 5      # the SG seizure: onset 03/10/2002 9:46, file_seizure_ind 73

    check_onset(PAT_NUM, SURF, SEIZURE_NUM)
