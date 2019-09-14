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

from scipy.io.wavfile import read

import librosa
import librosa.display
from librosa.core.constantq import cqt

import matplotlib.pyplot as plt


class Percussion(object):
    AcousticBassDrum = 35
    AcousticSnare = 38
    ClosedHiHat = 42


def generate_midi():

    # Create the MIDIFile Object with 1 track
    midi_file = MIDIFile(11)

    # Tracks are numbered from zero. Times are measured in beats.
    track = 0
    time = 0

    # Add track name and tempo.
    midi_file.addTrackName(track, time, "Sample Track")
    midi_file.addTempo(track, time, 120)

    # Add a note. addNote expects the following information:
    track = 10
    channel = 9
    pitch = 60
    time = 0
    duration = 1
    volume = 100

    # Now add the note.
    for i in range(12):
        midi_file.addNote(track, 0, pitch+i, time+i, duration, volume)
        #midi_file.addNote(track, channel, Percussion.ClosedHiHat, time + i, duration, volume)
    return midi_file


def convert_midi(midi_file):
    with open("output.mid", 'wb') as f_binary:
        midi_file.writeFile(f_binary)

    os.system("fluidsynth -F output_stereo.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 output.mid")
    os.system("sox output_stereo.wav output.wav channels 1")
    os.system("audacious output.wav")

    sample_rate, data = read("output.wav")
    print(data.shape)

    data = data.astype(float)
    data = (data + 0.5) / 32767.5

    if False:
        sr = 44100
        data = cqt(data, sr=sr, n_bins=100, bins_per_octave=12, hop_length=2048)
        print(data.shape)

        C = np.abs(data)
        librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr = sr, x_axis = 'time', y_axis = 'cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.tight_layout()
        plt.show()
        import IPython; IPython.embed()

    if True:
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
