#!/usr/bin/env python
"""
Datagen
"""

from __future__ import division, print_function

import argparse
import os
import random
import sys
import numpy as np

from midiutil.MidiFile import MIDIFile
import midi_constants

from scipy.io.wavfile import read

import librosa
import librosa.display
from librosa.core.constantq import cqt
import rainbow

import utils

import matplotlib.pyplot as plt


PERCUSSION_CHANNEL = 9


class Percussion(object):
    AcousticBassDrum = 35
    AcousticSnare = 38
    ClosedHiHat = 42


class Instrument(object):
    def __init__(self, instrument_code, track, channel):
        self.instrument_code = instrument_code
        self.track = track
        self.channel = channel


class Note(object):
    def __init__(self, instrument, pitch, t, duration):
        self.instrument = instrument
        self.pitch = pitch
        self.t = t
        self.duration = duration


class MidiFileWrapper(object):
    def __init__(self, tempo=60):
        # https://midiutil.readthedocs.io/en/1.2.1/class.html
        self.tempo = tempo
        self.midi_file = MIDIFile(numTracks=16)

        # Setup tempo
        self.midi_file.addTempo(
            track=0,  # For MIDI file type 1, the track is actually ignored
            time=0,
            tempo=tempo,
        )

        self.notes = []

    def add_note(self, instrument, pitch, t, duration, volume=100):
        self.midi_file.addNote(instrument.track, instrument.channel, pitch, t, duration, volume)
        self.notes.append(Note(
            instrument=instrument,
            pitch=pitch,
            t=t,
            duration=duration,
        ))

    def extract_groundtruth(self, raw_length, sample_rate, hop_length, lowest_note, highest_note, bins_per_note):
        print("Extracting ground truth for {} notes".format(len(self.notes)))
        raw_data = np.zeros((highest_note - lowest_note + 1, raw_length)).astype(np.int8)

        def compute_index(beat):
            t = beat * 60 / self.tempo
            return int(t * sample_rate)

        for note in self.notes:
            index_start = compute_index(note.t)
            index_end = compute_index(note.t + note.duration)
            # row = np.zeros(raw_length)
            # row[index_start:index_end] = 1

            assert note.pitch >= lowest_note
            assert note.pitch <= highest_note
            row_index = (note.pitch - lowest_note) * bins_per_note
            # raw_data[row_index, :] = np.max([raw_data[row_index, :], row], axis=0)
            raw_data[row_index, index_start:index_end] = 1

        # could be optimized, but probably not crucial:
        # https://stackoverflow.com/questions/15956309/averaging-over-every-n-elements-of-a-numpy-array#comment77786037_15956341
        group_means = [
            raw_data[:, i:i+hop_length].mean(axis=1)
            for i in range(raw_length)[::hop_length]
        ]
        groundtruth = np.stack(group_means, axis=1)
        return groundtruth


def setup_instruments(midi_file, instrument_codes):
    instruments = []
    i = 0
    for instrument_code in instrument_codes:
        if i == PERCUSSION_CHANNEL:
            continue
        midi_file.addProgramChange(tracknum=i, channel=i, time=0, program=instrument_code)
        instruments.append(Instrument(
            instrument_code=instrument_code,
            track=i,
            channel=i,
        ))
        i += 1
    return instruments


def setup_default_instruments(midi_file):
    # Setup instruments
    instruments = setup_instruments(midi_file, [
        midi_constants.PIANO,
        midi_constants.EPIANO1,
        midi_constants.NYLON_GUITAR,
        midi_constants.CLEAN_GUITAR,
        midi_constants.OVERDRIVEN_GUITAR,
        midi_constants.DISTORTION_GUITAR,
    ])
    return instruments


def generate_midi_chromatic_sweep():
    mfw = MidiFileWrapper(120)
    instruments = setup_default_instruments(mfw.midi_file)

    base_pitch = 60
    i = 1
    for instrument in instruments:
        for j in range(12):
            volume = 100
            duration = 1
            mfw.add_note(instrument, base_pitch+j, i, duration, volume)
            i += duration

    #midi_file.addNote(track, PERCUSSION_CHANNEL, Percussion.ClosedHiHat, time + i, duration, volume)
    return mfw


def generate_midi_random_single_notes():
    mfw = MidiFileWrapper(60)
    instruments = setup_default_instruments(mfw.midi_file)

    t = 0.0
    while t < 60:
        instrument = random.choice(instruments)
        volume = 100
        duration = np.random.uniform(0.075, 0.5)
        pitch = np.random.randint(60 - 24, 60 + 24)
        mfw.add_note(instrument, pitch, t, duration, volume)
        t += duration + np.random.uniform(0.0, 0.1)

    return mfw


def store_midi_and_wave(midi_file, path_midi, path_wave):
    print("Writing MIDI: {}".format(path_midi))
    with open(path_midi, 'wb') as f_binary:
        midi_file.writeFile(f_binary)

    print("Writing WAVE: {}".format(path_wave))
    os.system("fluidsynth -F /tmp/output_stereo.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 '{}'".format(path_midi))
    os.system("sox /tmp/output_stereo.wav '{}' channels 1".format(path_wave))


def read_wave(filename):
    sample_rate, wave_data = read(filename)

    wave_data = wave_data.astype(np.float32)
    wave_data = (wave_data + 0.5) / 32767.5

    return sample_rate, wave_data


def generate_dataset(mfw, base_path, audio_preview=False, use_cqt=True, interactive_plots=False):
    utils.ensure_parent_exists(base_path)

    path_midi = "{}.mid".format(base_path)
    path_wave = "{}.wav".format(base_path)
    store_midi_and_wave(mfw.midi_file, path_midi, path_wave)

    if audio_preview:
        os.system("audacious '{}' &".format(path_wave))

    sr, wave_data = read_wave(path_wave)

    if use_cqt:
        bins_per_note = 4
        bins_per_octave = 12 * bins_per_note
        n_octaves = 9
        n_bins = n_octaves * bins_per_octave
        hop_length = 512
        lowest_note_name = "C1"  # cqt default
        lowest_note_hz = librosa.note_to_hz(lowest_note_name)
        lowest_note_midi = librosa.note_to_midi(lowest_note_name)
        # https://librosa.github.io/librosa/generated/librosa.core.cqt.html
        C = cqt(
            wave_data,
            sr=sr,
            fmin=lowest_note_hz,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            filter_scale=1.0,
            #sparsity=0.0,
            tuning=0.0,     # we don't want automatic tuning estimation
        )

        #mag, phase = librosa.core.magphase(C)
        mag = np.abs(C)
        mag = mag.astype(np.float32)

        # 16th notes at 200 bpm are 800 notes/min = 13.3 notes/sec => note duration = 75 ms
        print("Sample rate: {}".format(sr))
        print("Hop duration: {:.1f} ms".format(hop_length / sr * 1000))
        print("Length audio: {:.1f} sec".format(len(wave_data) / sr))
        print("Shape audio:       {} [{}, {:.1f} MB]".format(
            wave_data.shape, wave_data.dtype, wave_data.nbytes / 1e6))
        print("Shape transformed: {} [{}, {:.1f} MB]".format(
            mag.shape, mag.dtype, mag.nbytes / 1e6))

        # Groundtruth extraction with same shape
        groundtruth = mfw.extract_groundtruth(
            raw_length=len(wave_data),
            sample_rate=sr,
            lowest_note=lowest_note_midi,
            highest_note=lowest_note_midi + n_octaves * bins_per_octave - 1,
            hop_length=hop_length,
            bins_per_note=4,
        )

        plot_dataset(C, groundtruth, base_path, sr, lowest_note_hz, hop_length, bins_per_octave, interactive_plots)

    else:
        np.save("wavedata.npy", data)
        os.system("nim -r c ./src/process_wave_data wavedata.npy")
        data = np.load("wavedata_preprocessed.npy")
        data = data[::-1, :]

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
        plt.imshow(data, aspect='auto')
        plt.show()


def plot_dataset(C, groundtruth, base_path, sr, lowest_note_hz, hop_length, bins_per_octave, interactive_plots):
    mag = np.abs(C)
    db = librosa.amplitude_to_db(mag, ref=np.max)

    def raw_plot(data, filename, cmap=None):
        dpi = 72
        height, width = np.array(data.shape, dtype=float) / dpi

        fig = plt.figure(figsize=(width, height), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        if cmap is None:
            cmap = librosa.display.cmap(data)

        ax.imshow(data[::-1, :], interpolation='none', cmap=cmap)
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    def plot_with_axis_annotations(data, ax=None, with_color_bar=True):
        librosa.display.specshow(
            data,
            sr=sr,
            fmin=lowest_note_hz,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            x_axis='time',
            y_axis='cqt_note',
            ax=ax,
        )

    raw_plot(
        groundtruth,
        "{}_0_gt.png".format(base_path))

    raw_plot(
        db,
        "{}_1_db.png".format(base_path))

    raw_plot(
        db[:, 1:] - db[:, :-1],
        "{}_delta_t.png".format(base_path), cmap="magma")

    raw_plot(
        db[1:, :] - db[:-1, :],
        "{}_delta_b.png".format(base_path), cmap="magma")

    # Value distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    data = mag.flatten()
    axes[0, 0].hist(data, bins=200)
    values, base = np.histogram(data, bins=200)
    cumulative = np.cumsum(values)
    axes[1, 0].plot(base[:-1], cumulative / len(data), c='blue')

    data = db.flatten()
    axes[0, 1].hist(data, bins=200)
    values, base = np.histogram(data, bins=200)
    cumulative = np.cumsum(values)
    axes[1, 1].plot(base[:-1], cumulative / len(data), c='blue')

    fig.suptitle(
        "min(mag) = {}    "
        "max(mag) = {}    "
        "min(db) = {}    "
        "max(db) = {}    ".format(mag.min(), mag.max(), db.min(), db.max()))
    fig.tight_layout()
    plt.savefig("{}_value_distribution.png".format(base_path))
    plt.close(fig)

    # Rainbow
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    rainbow.plot_rainbow(ax, C)
    fig.tight_layout()
    plt.savefig("{}_rainbow.png".format(base_path))

    if interactive_plots:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)
        plot_with_axis_annotations(librosa.amplitude_to_db(mag, ref=np.max), axes[0])
        plot_with_axis_annotations(groundtruth, axes[1])

        # We need to fetch the QuadMesh instance to pass to colorbar
        plt.colorbar(axes[0].get_children()[0], ax=axes[0], format='%+2.0f dB')
        plt.colorbar(axes[1].get_children()[0], ax=axes[1])
        fig.tight_layout()

    if interactive_plots:
        plt.show()


def main():
    training_data_dir = utils.path_rel_to_base("data")
    output_path = os.path.join(training_data_dir, "train", "dataset_001")

    midi_file = generate_midi_chromatic_sweep()
    #midi_file = generate_midi_random_single_notes()

    generate_dataset(midi_file, output_path, audio_preview=False)


if __name__ == "__main__":
    main()
