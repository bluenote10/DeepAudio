"""
Adapted from:
https://gist.github.com/jesseengel/e223622e255bd5b8c9130407397a0494
"""

from __future__ import print_function


import os

import librosa
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from scipy.io.wavfile import read as readwav


# Plotting functions
cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

         'alpha': ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
         }

my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
plt.register_cmap(cmap=my_mask)


def plot_rainbow(ax, C):
    mag, phase = librosa.core.magphase(C)

    phase_angle = np.angle(phase)
    phase_unwrapped = np.unwrap(phase_angle)
    dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    #dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
    dphase = np.concatenate([np.zeros((dphase.shape[0], 1)), dphase], axis=1) / np.pi

    # mag = (librosa.logamplitude(mag ** 2, amin=1e-13, top_db=peak, ref_power=np.max) / peak) + 1
    mag = librosa.power_to_db(mag ** 2, amin=1e-13, ref=np.max)

    #ax.matshow(phase_angle[::-1, :], cmap=plt.cm.rainbow, aspect="auto")
    ax.matshow(dphase[::-1, :], cmap=plt.cm.rainbow, aspect="auto")
    ax.matshow(mag[::-1, :], cmap=my_mask, aspect="auto")


def note_specgram(path, ax, use_cqt=True):

    # Add several samples together
    if isinstance(path, list):
        for i, f in enumerate(path):
            sr, a = readwav(f)
            audio = a if i == 0 else a + audio
    # Load one sample
    else:
        sr, audio = readwav(path)

    audio = audio.astype(np.float32)

    # Constants
    hop_length = 256
    if use_cqt:
        over_sample = 4
        res_factor = 1.0    # 0.8
        octaves = 6
        notes_per_octave = 12
        C = librosa.cqt(audio, sr=sr, hop_length=hop_length,
                        bins_per_octave=int(notes_per_octave * over_sample),
                        n_bins=int(octaves * notes_per_octave * over_sample),
                        filter_scale=res_factor,
                        fmin=librosa.note_to_hz('C3'))
    else:
        n_fft = 512
        C = librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)

    plot_rainbow(ax, C)


def plot_notes(list_of_paths, rows=2, cols=4, col_labels=[], row_labels=[],
               use_cqt=True, peak=70.0):
    """Build a CQT rowsXcols.
    """

    N = len(list_of_paths)
    assert N == rows * cols

    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.07, right=0.95, bottom=0.05, top=0.93, wspace=0.05, hspace=0.1,
    )

    if not isinstance(axes, list):
        axes = [axes]

    #   fig = plt.figure(figsize=(18, N * 1.25))
    for i, path in enumerate(list_of_paths):
        row = i / cols
        col = i % cols
        if rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        note_specgram(path, ax, use_cqt)

        """
        ax.set_xticks([]);
        ax.set_yticks([])
        if col == 0 and row_labels:
            ax.set_ylabel(row_labels[row])
        if row == rows - 1 and col_labels:
            ax.set_xlabel(col_labels[col])
        """

    plt.show()
