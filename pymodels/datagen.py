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


def generate_midi():

    # Create the MIDIFile Object with 1 track
    midi_file = MIDIFile(1)

    # Tracks are numbered from zero. Times are measured in beats.
    track = 0
    time = 0

    # Add track name and tempo.
    midi_file.addTrackName(track, time, "Sample Track")
    midi_file.addTempo(track, time, 120)

    # Add a note. addNote expects the following information:
    track = 0
    channel = 0
    pitch = 60
    time = 0
    duration = 1
    volume = 100

    # Now add the note.
    midi_file.addNote(track, channel, pitch, time, duration, volume)
    return midi_file


def convert_midi(midi_file):
    with open("output.mid", 'wb') as f_binary:
        midi_file.writeFile(f_binary)

    os.system("fluidsynth -F output_stereo.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 output.mid")
    os.system("sox output_stereo.wav output.wav channels 1")
    os.system("audacious output.wav")

    sample_rate, data = read("output.wav")

    data = data.astype(float)
    data = (data + 0.5) / 32767.5

    np.save("wavedata.npy", data)
    import IPython; IPython.embed()


def main():
    midi_file = generate_midi()
    convert_midi(midi_file)


if __name__ == "__main__":
    main()
