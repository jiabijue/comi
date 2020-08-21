import os
import sys
import glob
import csv
import random
import multiprocessing
from tqdm import tqdm

from tokenization_by_time import MidiTokenizer


if __name__ == '__main__':
    dataset_name = sys.argv[1]

    txt_dir = './data/{}/txt'.format(dataset_name)

    os.makedirs(os.path.join(txt_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(txt_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(txt_dir, 'test'), exist_ok=True)

    tokenizer = MidiTokenizer()
    tokenizer.vocab.save()

    if dataset_name == 'midicn':
        midi_dir = './data/midicn/cleaned_midi'
        midi_files = glob.glob(midi_dir + '/*.mid*')
        random.shuffle(midi_files)

        train_split = int(0.8 * len(midi_files))
        valid_split = int(0.9 * len(midi_files))

        train_files = midi_files[:train_split]
        valid_files = midi_files[train_split:valid_split]
        test_files = midi_files[valid_split:]

        label_dict = dict()
        with open('./data/midicn/control_codes.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                label_dict[row[1]] = int(row[0])

        ctrl_codes = ['<animated_cartoon>', '<lovely>', '<hk_tw>', '<classical>',
                      '<jazz>', '<usa_euro_pop>', '<christmas>', '<games>']

        def _save_txt(midi_f, subset):
            fname = str(os.path.basename(midi_f).split('.')[0])

            try:
                tokens = tokenizer.tokenize(midi_f)
            except:
                return

            if tokens is not None:
                with open(os.path.join(txt_dir, subset, fname + '.txt'), 'w') as f:
                    f.write('\n'.join([ctrl_codes[label_dict[os.path.basename(midi_f)]]] + tokens))  # add control code

        for subset, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
            def _task(x):
                _save_txt(x, subset)

            with multiprocessing.Pool(8) as p:
                r = list(tqdm(p.imap(_task, files), total=len(files)))

