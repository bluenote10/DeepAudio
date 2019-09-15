#!/usr/bin/env python
"""
Datagen
"""

from __future__ import print_function

import argparse
import os
import sys
import numpy as np

from midiutil.MidiFile import MIDIFile
import midi_constants

from scipy.io.wavfile import read

import librosa
import librosa.display
from librosa.core.constantq import cqt
from rainbow import plot_notes

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


def generate_midi():

    # https://midiutil.readthedocs.io/en/1.2.1/class.html
    midi_file = MIDIFile(numTracks=16)

    # Setup tempo
    tempo = 120     # * 4
    midi_file.addTempo(
        track=0,    # For MIDI file type 1, the track is actually ignored
        time=0,
        tempo=tempo,
    )

    # Setup instruments
    instruments = setup_instruments(midi_file, [
        midi_constants.PIANO,
        midi_constants.EPIANO1,
        midi_constants.NYLON_GUITAR,
        midi_constants.CLEAN_GUITAR,
        midi_constants.OVERDRIVEN_GUITAR,
        midi_constants.OVERDRIVEN_GUITAR,
        midi_constants.DISTORTION_GUITAR,
    ])

    base_pitch = 60

    # Now add the note.
    i = 1
    for instrument in instruments:
        for j in range(12):
            volume = 100
            duration = 1
            midi_file.addNote(instrument.track, instrument.channel, base_pitch+j, i, duration, volume)
            i += duration

    #midi_file.addNote(track, PERCUSSION_CHANNEL, Percussion.ClosedHiHat, time + i, duration, volume)

    return midi_file


def convert_midi(midi_file):
    with open("output.mid", 'wb') as f_binary:
        midi_file.writeFile(f_binary)

    os.system("fluidsynth -F output_stereo.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 output.mid")
    os.system("sox output_stereo.wav output.wav channels 1")
    #os.system("audacious output.wav")

    plot_notes(["output.wav"], rows=1, cols=1)

    sample_rate, data = read("output.wav")
    print(data.shape)

    data = data.astype(float)
    data = (data + 0.5) / 32767.5

    if True:
        sr = 44100
        bins_per_octave = 48
        n_octaves = 9
        n_bins = n_octaves * bins_per_octave
        # https://librosa.github.io/librosa/generated/librosa.core.cqt.html
        data = cqt(
            data,
            sr=sr,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=512,
            filter_scale=1.0,
            #sparsity=0.0,
            tuning=0.0,     # we don't want automatic tuning estimation
        )
        print(data.shape)

        C = np.abs(data)
        librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.tight_layout()
        plt.show()
        #import IPython; IPython.embed()
        sys.exit(0)

    if False:
        sample_rate, data = read("output.wav")
        print(data.shape)

        data = data.astype(float)
        data = (data + 0.5) / 32767.5

        np.save("wavedata.npy", data)
        #os.system("./src/process_wave_data wavedata.npy")
        os.system("nim -r c ./src/process_wave_data wavedata.npy")
        data = np.load("wavedata_preprocessed.npy")
        data = data[::-1, :]

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
    plt.imshow(data, aspect='auto')
    plt.show()
    import IPython; IPython.embed()


def main():
    midi_file = generate_midi()
    convert_midi(midi_file)


if __name__ == "__main__":
    main()
