# COMI

Codes are tested on:
- python 3.7
- pytorch 1.4.0
- pretty_midi 0.2.9
- mido 1.2.9


## Data Preparation
data processing-related files:
- data_config.py
- data_cleaning.py
- tokenization_by_time.py
- preprocess.py

First, the midi data is put under `./data/midicn/midi/` 

Then, do some data filtering and cleaning. The cleaned midi files are put 
under `./data/midicn/cleaned_midi/`
```
python data_cleaning.py midicn
```

Then, pre-process each midi file and save as a txt file; at the same time,
the `./data/vocab.txt` is created. Saved txt files are separated as three 
sub-sets (train, val, test) and are put under `./data/midicn/txt/`.
This step is to represent a midi as a token sequence using the vocabulary
that we've defined.
```
python preprocess.py midicn
```


## Train
