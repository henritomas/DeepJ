import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch
import music21.midi as midi21


import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from midicom import *

#In ubuntu you must manually set the default program to open *Path by specifying the path to that program
music21.environment.set('graphicsPath', '/usr/bin/eog') #Eye of Gnome, Ubuntu's default image viewer
music21.environment.set('midiPath', '/usr/bin/timidity') #timidity, installed midi player

from music21 import stream
from music21 import roman

import os 
import pickle
import pprint
import gensim, logging

def main():
    #print_parts_contour(my_midi)
    #red_midi = harmonic_reduction(base_midi)
    #print(red_midi[:10])

    #save_w2v_train()
    sentences = load_w2v_train()
    model = gensim.models.Word2Vec(sentences, min_count=2, window=4, size=50, iter=100)

    print("List of chords found:")
    print(model.wv.vocab.keys())
    print("Number of chords considered by model: {0}".format(len(model.wv.vocab)))

    midA = "./word2vec/data/midi2017_(1).mid"
    midB = "./word2vec/data/midi2017_(100).mid"
    mid0 = "/home/henri/Downloads/output_0.mid"
    mid1 = "/home/henri/Downloads/output_1.mid"
    mid2 = "/home/henri/Downloads/output_2.mid"
    mid3 = "/home/henri/Downloads/midireee/output_3(1).mid"
    res = calculate_similarity_aux(model, mid0, [mid1, mid2, mid3], threshold = -1)
    print(res)
    #get_related_chords(model, 'I')
    #get_related_chords(model, 'iv')
    #get_related_chords(model, 'V')
    #get_chord_similarity(model, 'iv', 'I')

def vectorize_harmony(model, harmonic_reduction):
    # Gets the model vector values for each chord from the reduction.
    word_vecs = []
    for word in harmonic_reduction:
        try:
            vec = model[word]
            word_vecs.append(vec)
        except KeyError:
            # Ignore, if the word doesn't exist in the vocabulary
            pass
    
    # Assuming that document vector is the mean of all the word vectors.
    return np.mean(word_vecs, axis=0)

def cosine_similarity(vecA, vecB):
    # Find the similarity between two vectors based on the dot product.
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    if np.isnan(np.sum(csim)):
        return 0
    
    return csim

def calculate_similarity_aux(model, source_name, target_names=[], threshold=0):
    source_midi = open_midi(source_name, True)
    source_harmo = harmonic_reduction(source_midi)
    source_vec = vectorize_harmony(model, source_harmo)    
    results = []
    for name in target_names:
        print(f"loading {name}")
        target_midi = open_midi(name, True)
        target_harmo = harmonic_reduction(target_midi)
        if (len(target_harmo) == 0):
            continue
            
        target_vec = vectorize_harmony(model, target_harmo)       
        sim_score = cosine_similarity(source_vec, target_vec)
        if sim_score > threshold:
            results.append({
                'score' : sim_score,
                'name' : name
            })
                
    # Sort results by score in desc order
    results.sort(key=lambda k : k['score'] , reverse=True)
    return results

def get_related_chords(model, token, topn=100): #default top 3 most similar words
    print("Similar chords with " + token)
    for word, similarity in model.wv.most_similar(positive=[token], topn=topn):
        print (word, round(similarity, 5))

def get_chord_similarity(model, chordA, chordB):
    print("Similarity between {0} and {1}: {2}".format(
        chordA, chordB, model.wv.similarity(chordA, chordB)))

def load_w2v_train():
    load_dir = "./word2vec/cache/"
    sentences = []
    for filename in sorted(os.listdir(load_dir)):
        if filename.endswith('pickle'):
            #pickle
            infile = open(load_dir+filename, 'rb')
            red_midi = pickle.load(infile)
            infile.close()  
            sentences.append(red_midi)
    return sentences

def save_w2v_train():
    directory = "./word2vec/data/"
    save_dir = "./word2vec/cache/"
    for filename in sorted(os.listdir(directory)):
        print(filename)
        savename = filename[:-4]+"pickle"
        if savename in os.listdir(save_dir):
            continue
        elif filename.endswith('.mid'):
            base_midi = open_midi(directory + filename, True)
            red_midi = harmonic_reduction(base_midi)

            #pickle
            outfile = open(save_dir+savename, 'wb')
            pickle.dump(red_midi, outfile)
            outfile.close()  

def note_count(measure, count_dict):
    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        # All notes have the same length of its chord parent.
        note_length = chord.quarterLength
        for note in chord.pitches:          
            # If note is "C5", note.name is "C". We use "C5"
            # style to be able to detect more precise inversions.
            note_name = str(note) 
            if (bass_note is None or bass_note.ps > note.ps):
                bass_note = note
                
            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length
        
    return bass_note

def simplify_roman_name(roman_numeral):
    # Chords can get nasty names as "bII#86#6#5",
    # in this method we try to simplify names, even if it ends in
    # a different chord to reduce the chord vocabulary and display
    # chord function clearer.
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()
    
    # Checking valid inversions.
    if ((roman_numeral.isTriad() and inversion < 3) or
            (inversion < 4 and
                 (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()
        
    if (inversion_name is not None):
        ret = ret + str(inversion_name)
        
    elif (roman_numeral.isDominantSeventh()): ret = ret + "M7"
    elif (roman_numeral.isDiminishedSeventh()): ret = ret + "o7"
    return ret

def harmonic_reduction(midi_file):
    ret = []
    temp_midi = stream.Score()
    temp_midi_chords = midi_file.chordify()
    temp_midi.insert(0, temp_midi_chords)    
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 4   
    for m in temp_midi_chords.measures(0, None): # None = get all measures.
        if (type(m) != stream.Measure):
            continue
        
        # Here we count all notes length in each measure,
        # get the most frequent ones and try to create a chord with them.
        count_dict = dict()
        bass_note = note_count(m, count_dict)
        if (len(count_dict) < 1):
            ret.append("-") # Empty measure
            continue
        
        sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)
        
        # Convert the chord to the functional roman representation
        # to make its information independent of the music key.
        roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
        ret.append(simplify_roman_name(roman_numeral))
        
    return ret

if __name__ == "__main__":
    main()
