# DeepJ-Mood: A variant of the DeepJ model for mood-specific music generation

## Sample Outputs
Listen to our best outputs at [sound cloud](https://soundcloud.com/gabriel-perez-90/sets/deep-learning-based-on-music)

A COE 197-Z / EE 298-Z Final Project. 
* [Github link](https://github.com/henritomas/DeepJ)
* Documentation: [ipynb - Jupyter notebook](https://github.com/henritomas/DeepJ/blob/icsc/documentation.ipynb)
* View our [Google Slides](https://docs.google.com/presentation/d/19dG5vqjuzJXPAvmT2QsxvO6G5LpGiHpg1ghIl1e3tUI/edit?usp=sharing)
* Try generating some samples at [Colab](https://colab.research.google.com/drive/1kTtIm-1eqqUgfHzycftgRqQLA-vsYOKo)

Main References ad Code bases:
[Calclavia - DeepJ](https://github.com/calclavia/DeepJ)
[MIDI Music Data Extraction with Music21](https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21)

## Usage
To train a new model, run the following command:
```
python train.py
```

To generate music, run the following command:
```
python generate.py
```
Use the help command to see CLI arguments:
```
python generate.py --help
```

# Piano2Vec : A word2vec model trained on the harmonic reduction of 140 piano midi files from MAESTRO Dataset
To use the word2vec model, run with word2vec folder in the same directory
```
python midi-similarity.py
```
