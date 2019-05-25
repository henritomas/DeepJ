# DeepJ-Mood: A variant of the DeepJ model for mood-specific music generation

## Sample Outputs
Listen to our best outputs at [sound cloud](https://soundcloud.com/gabriel-perez-90/sets/deep-learning-based-on-music)

A COE 197-Z / EE 298-Z Final Project. 
[github link](https://github.com/henritomas/DeepJ)
[ipynb link](https://github.com/henritomas/DeepJ/blob/icsc/documentation.ipynb)
Try generating some samples at [Colab](https://colab.research.google.com/drive/1kTtIm-1eqqUgfHzycftgRqQLA-vsYOKo)

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
