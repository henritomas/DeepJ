import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from music21 import converter, corpus, instrument, midi, note, chord, pitch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

mid = "Romantic 1.mid"
def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]          

    return midi.translate.midiFileToStream(mf)

base_midi = open_midi(mid, True)
base_midi.plot('histogram', 'pitchClass', 'count')