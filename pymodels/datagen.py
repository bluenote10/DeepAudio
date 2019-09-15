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

import matplotlib.pyplot as plt


PERCUSSION_CHANNEL = 9


class Percussion(object):
    AcousticBassDrum = 35
    AcousticSnare = 38
    ClosedHiHat = 42


class Instrument(object):
    def __init__(self, track, channel):
        self.track = track
        self.channel = channel


def setup_instruments(midi_file, instrument_codes):
    instruments = []
    i = 0
    for instrument_code in instrument_codes:
        if i == PERCUSSION_CHANNEL:
            continue
        midi_file.addProgramChange(tracknum=i, channel=i, time=0, program=instrument_code)
        instruments.append(Instrument(track=i, channel=i))
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


def init_midi_file(tempo=60):
    # https://midiutil.readthedocs.io/en/1.2.1/class.html
    midi_file = MIDIFile(numTracks=16)

    # Setup tempo
    midi_file.addTempo(
        track=0,    # For MIDI file type 1, the track is actually ignored
        time=0,
        tempo=tempo,
    )
    return midi_file


def generate_midi_chromatic_sweep():
    midi_file = init_midi_file(120)
    instruments = setup_default_instruments(midi_file)

    base_pitch = 60
    i = 1
    for instrument in instruments:
        for j in range(12):
            volume = 100
            duration = 1
            midi_file.addNote(instrument.track, instrument.channel, base_pitch+j, i, duration, volume)
            i += duration

    #midi_file.addNote(track, PERCUSSION_CHANNEL, Percussion.ClosedHiHat, time + i, duration, volume)
    return midi_file


def generate_midi_random_single_notes():
    midi_file = init_midi_file(60)
    instruments = setup_default_instruments(midi_file)

    t = 0.0
    while t < 60:
        instrument = random.choice(instruments)
        volume = 100
        duration = np.random.uniform(0.075, 0.5)
        pitch = np.random.randint(60 - 24, 60 + 24)
        midi_file.addNote(instrument.track, instrument.channel, pitch, t, duration, volume)
        t += duration + np.random.uniform(0.0, 0.1)

    return midi_file


def read_wave(filename):
    sample_rate, wave_data = read(filename)

    wave_data = wave_data.astype(np.float32)
    wave_data = (wave_data + 0.5) / 32767.5

    return sample_rate, wave_data


def convert_midi(midi_file, audio_preview=False, use_cqt=True):
    with open("output.mid", 'wb') as f_binary:
        midi_file.writeFile(f_binary)

    os.system("fluidsynth -F output_stereo.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 output.mid")
    os.system("sox output_stereo.wav output.wav channels 1")

    if audio_preview:
        os.system("audacious output.wav &")

    sr, wave_data = read_wave("output.wav")

    if use_cqt:
        bins_per_octave = 48
        n_octaves = 9
        n_bins = n_octaves * bins_per_octave
        hop_length = 512
        fmin = librosa.note_to_hz('C1')     # cqt default
        # https://librosa.github.io/librosa/generated/librosa.core.cqt.html
        C = cqt(
            wave_data,
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            filter_scale=1.0,
            #sparsity=0.0,
            tuning=0.0,     # we don't want automatic tuning estimation
        )
        mag, phase = librosa.core.magphase(C)
        mag = mag.astype(np.float32)
        # 16th notes at 200 bpm are 800 notes/min = 13.3 notes/sec => note duration = 75 ms
        print("Sample rate: {}".format(sr))
        print("Hop duration: {:.1f} ms".format(hop_length / sr * 1000))
        print("Length audio: {:.1f} sec".format(len(wave_data) / sr))
        print("Shape audio:       {} [{}, {:.1f} MB]".format(
            wave_data.shape, wave_data.dtype, wave_data.nbytes / 1e6))
        print("Shape transformed: {} [{}, {:.1f} MB]".format(
            mag.shape, mag.dtype, mag.nbytes / 1e6))

        plots = ["default", "rainbow"]
        for plot in plots:
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            if plot == "rainbow":
                rainbow.plot_rainbow(ax, C)
            else:
                librosa.display.specshow(
                    librosa.amplitude_to_db(mag, ref=np.max),
                    sr=sr,
                    fmin=fmin,
                    hop_length=hop_length,
                    bins_per_octave=bins_per_octave,
                    x_axis='time',
                    y_axis='cqt_note',
                )
                plt.colorbar(format='%+2.0f dB')
            fig.tight_layout()

        if len(plots) > 0:
            plt.show()

    else:
        np.save("wavedata.npy", data)
        os.system("nim -r c ./src/process_wave_data wavedata.npy")
        data = np.load("wavedata_preprocessed.npy")
        data = data[::-1, :]

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
        plt.imshow(data, aspect='auto')
        plt.show()


def main():
    #midi_file = generate_midi_chromatic_sweep()
    midi_file = generate_midi_random_single_notes()
    convert_midi(midi_file, audio_preview=True)


if __name__ == "__main__":
    main()
