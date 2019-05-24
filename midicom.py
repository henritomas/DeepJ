import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch
import music21.midi as midi21
from music21 import stream
from music21 import roman

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from math import sqrt

import midi as pmidi
import midi.timeresolver as tres

from constants import *

#In ubuntu you must manually set the default program to open *Path by specifying the path to that program
music21.environment.set('graphicsPath', '/usr/bin/eog') #Eye of Gnome, Ubuntu's default image viewer
music21.environment.set('midiPath', '/usr/bin/timidity') #timidity, installed midi player

def open_midi(midi_path, remove_drums=True):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi21.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]          

    return midi.translate.midiFileToStream(mf)

def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
    
    return ret, parent_element

def print_parts_contour(midi):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minPitch = pitch.Pitch('C10').ps
    maxPitch = 0
    xMax = 0
    
    # Drawing notes.
    for i in range(len(midi.parts)):
        top = midi.parts[i].flat.notes                  
        y, parent_element = extract_notes(top)
        if (len(y) < 1): continue
            
        x = [n.offset for n in parent_element]
        ax.scatter(x, y, alpha=0.6, s=7)
        
        aux = min(y)
        if (aux < minPitch): minPitch = aux
            
        aux = max(y)
        if (aux > maxPitch): maxPitch = aux
            
        aux = max(x)
        if (aux > xMax): xMax = aux
    
    for i in range(1, 10):
        linePitch = pitch.Pitch('C{0}'.format(i)).ps
        if (linePitch > minPitch and linePitch < maxPitch):
            ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))            

    plt.ylabel("Note index (each octave has 12 notes)")
    plt.xlabel("Number of quarter notes (beats)")
    plt.title('Voices motion approximation, each color is a different instrument, red lines show each octave')
    plt.show()

def print_contour(midifile):
    temp_midi_chords = open_midi(midifile, True).chordify()
    temp_midi = stream.Score()
    temp_midi.insert(0, temp_midi_chords)

    #Printing merged tracks.
    print_parts_contour(temp_midi)

def key_coco(midifile):
    base_midi = open_midi(midifile, True)

    timeSignature = base_midi.getTimeSignatures()[0]
    music_analysis = base_midi.analyze('key')
    #print("Music time signature: {0}/{1}".format(timeSignature.beatCount, timeSignature.denominator))
    #print("Expected music key: {0}".format(music_analysis))
    #print("Music key confidence: {0}".format(music_analysis.correlationCoefficient))
    key_coco = {music_analysis: 1}
    max_key = music_analysis.correlationCoefficient
    #print("Other music key alternatives:")
    for analysis in music_analysis.alternateInterpretations:
        if (1):
            key_coco[analysis] = analysis.correlationCoefficient/max_key

    return key_coco

def rmse_coco(midiA, midiB):
    cocoA = key_coco(midiA)
    cocoB = key_coco(midiB)

    sum = 0
    for key in cocoA.keys():
        sum += (cocoA[key] - cocoB[key])**2
    rmse = sqrt(sum / len(cocoA))
    return rmse

def mae_coco(midiA, midiB):
    cocoA = key_coco(midiA)
    cocoB = key_coco(midiB)

    sum = 0
    for key in cocoA.keys():
        sum += abs(cocoA[key] - cocoB[key])
    mae = sum / len(cocoA)
    return mae

#Training data is measured in milliseconds, so millis=True if training data
#millis=False if output data, which is in seconds
def note_rate(mid, millis):
    base_midi = open_midi(mid, True)
    pattern = pmidi.read_midifile(mid)
    pattern.make_ticks_abs()
    time_resolver = tres.TimeResolver(pattern)
    total_ms = 0
    for track in pattern:
        for event in track:
            name = event.name
            isNote = "Note" in name
            if isNote:
                pitch = event.pitch
                velocity = event.velocity
            else:
                pitch = 0
                velocity = 0
            tick = event.tick
            milliseconds = time_resolver.tick2ms(tick)
            if milliseconds > total_ms and isNote:
                total_ms = milliseconds
            #print(f"event {name} with MIDI tick {tick} and pitch,velocity {pitch},{velocity} happens after {milliseconds} milliseconds.")
    if millis:
        seconds = total_ms/1000
    else:
        seconds = total_ms
    num_notes = len(base_midi.flat.notes)
    return num_notes/seconds

def note_usage(mid):
    """
    ratio of used_notes / total_notes (piano)
    """
    used_notes = []
    deepj_notes = MAX_NOTE - MIN_NOTE
    piano_notes = 108 - 21 + 1

    base_midi = open_midi(mid, True)
    pattern = pmidi.read_midifile(mid)
    pattern.make_ticks_abs()
    time_resolver = tres.TimeResolver(pattern)

    mnc = 0
    prev_pitch = 0
    for track in pattern:
        for event in track:
            name = event.name
            if ("Note On" in name) and (event.velocity != 0):
                pitch = event.pitch
                if prev_pitch:
                    mnc += abs(pitch - prev_pitch) 
                prev_pitch = pitch
                if pitch not in used_notes:
                    used_notes.append(pitch)
    num_notes = len(base_midi.flat.notes)
    mnc =  mnc / num_notes   #mean note change
    nur = len(used_notes)/piano_notes #note usage ratio
    return nur, mnc

def pitch_frame(mid):
    """
    dictionary with count of pitches used and unused
    """

    base_midi = open_midi(mid, True)
    pitch = base_midi.flat.pitches
    #.midi retrieves attribute midi number (e.g. 65) from pitch name (e.g. F4)
    pitch_list = [str(p.midi) for p in pitch] 
    #print(pitch_list)
    #print(len(pitch_list))

    max_count = 0 #for normalization
    pitch_frame = {}
    for note in range(MIN_NOTE, MAX_NOTE):
        count = pitch_list.count(str(note))
        pitch_frame[note] = count

        if count > max_count:
            max_count = count

    #Max-Normalizad pitch frame
    pitch_frame_norm = {}
    for key in pitch_frame.keys():
        pitch_frame_norm[key] = pitch_frame[key]/max_count

    return pitch_frame, pitch_frame_norm

""" Pitch Distribution Distance functions"""
def rmse_pdd(mid1, mid2):
    _,frame1 = pitch_frame(mid1)
    _,frame2 = pitch_frame(mid2)

    sum = 0
    for key in frame1.keys():
        sum += (frame1[key] - frame2[key])**2
    rmse = sqrt(sum / len(frame1))
    return rmse    

        

if __name__ == "__main__":
    
    midA = "./data/happy/bach_846.mid"
    midB = "./data/sad/bor_ps2.mid"
    mid0 = "/home/henri/Downloads/output_0.mid"
    mid1 = "/home/henri/Downloads/output_1.mid"
    mid2 = "/home/henri/Downloads/output_2.mid"
    mid3 = "/home/henri/Downloads/output_3.mid"
    #base_midi = open_midi(midC, True)
    #print_contour(midC)
    #print(base_midi.flat.elements)
    #base_midi.plot('histogram', 'pitchClass', 'count')
    #base_midi.plot('scatter', 'offset', 'pitchClass')
    #base_midi.plot('pianoroll')
    #base_midi.plot('histogram', 'pitch')
    #print(rmse_coco(midA, midB))
    #print(mae_coco(midA, midB))
    #print(note_rate(midB, True))
    #print(note_rate(midC, False))
    #print(note_usage(midC))
    #print(rmse_pdd(mid0, mid1))
    print_contour(midA)