import os
from bisect import bisect_left

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pretty_midi

from data_config import CTRL_TOKENS, INST_ENSEMBLE_SIZE, \
    DEFAULT_PITCH_RANGE, DEFAULT_VELOCITY_RANGE, QUANTIZE_VELOCITY, \
    DEFAULT_TEMPO, DEFAULT_RESOLUTION

"""
each note:
type    note    duration    velocity
<sep>
"""
INSTRUMENT_TOKENS = [f'i{i}' for i in range(INST_ENSEMBLE_SIZE)] + [f'i0d']
NOTE_TOKENS = [f'n{i}' for i in DEFAULT_PITCH_RANGE]
if QUANTIZE_VELOCITY:
    VELOCITY_TOKENS = ["v25", "v35", "v45", "v55", "v65", "v75", "v85", "v95", "v105"]
else:
    VELOCITY_TOKENS = [f'v{i}' for i in DEFAULT_VELOCITY_RANGE]

SEP_TOKEN = '<sep>'
END_TOKEN = '<eos>'
UNIT = 8  # e.g., ticks_per_beat=220, UNIT=8, then 1 time-step represents 60 ticks (ticks_per_beat/UNIT) 即把一个beat又切分成UNIT份作为一个时间单位

"""
duration: ticks_per_beat / note_duration
0.25    whole note
0.375   1.5 whole note
0.5     half note
0.75    1.5 half note
1       quarter note
1.5     1.5 quarter note
2       eighth note
3       1.5 eighth note
4       16th note
6       1.5 16th note
8       32th note
12      1.5 32th note
"""
DURATION_NOTES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DURATION_TOKENS = ["whole_note", "half_note", "quarter_note", "eth_note", "sth_note", "t2h_note"]
DURATION_TO_WORD = dict(zip(DURATION_NOTES, DURATION_TOKENS))
WORD_TO_DURATION = dict(zip(DURATION_TOKENS, DURATION_NOTES))

CAPTURE_MSG = ["note_on", "note_off", "program_change"]
"""
note_on     channel:    0 - 15      9 is reserved for percussion instruments; 0-8, 10-15 for others
            note:       0 - 127     piano key number. 69 => Middle-A note. For percussion instruments it is assigned to
                                    the instrument number
            velocity    0 - 127     the volume of force.
note_off    channel:    0 - 15      specify which note ends
"""

INSTRUMENT_FAMILY = ["piano", "organ", "guitar", "bass", "strings", "ensemble", "brass", "reed", "pipe", "lead", "pad",
                     "effects", "ethnic", "percussive"]
INSTRUMENT_DEFAULT_NUM = [0, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112]
DEFUALT_INSTRUMENT = dict(zip(INSTRUMENT_FAMILY, INSTRUMENT_DEFAULT_NUM))


class Vocabulary(object):
    def __init__(self):
        self.all_words = CTRL_TOKENS + [SEP_TOKEN] + [END_TOKEN] + INSTRUMENT_TOKENS + \
                         NOTE_TOKENS + DURATION_TOKENS + VELOCITY_TOKENS
        self.word2idx = {v: k for k, v in enumerate(self.all_words)}
        self.idx2word = {k: v for k, v in enumerate(self.all_words)}

    def __len__(self):
        return len(self.all_words)

    def save(self):
        with open('./data/vocab.txt', 'w') as f:
            f.write('\n'.join(self.all_words))


def take_closest(_list: list, num):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(_list, num)
    if pos == 0:
        return _list[0]
    if pos == len(_list):
        return _list[-1]
    before = _list[pos - 1]
    after = _list[pos]
    if after - num < num - before:
        return after
    else:
        return before


class Note(object):
    def __init__(self, note, velocity, start_time, instrument_type, channel, ticks_per_beat):
        self._note = note
        self._velocity = velocity
        self._start_time = start_time
        self._end_time = 0
        self._instrument_type = instrument_type
        self._ticks_per_beat = ticks_per_beat
        self._channel = channel

    @property
    def is_drum(self):
        return self._channel == 9

    @property
    def note(self):
        return self._note

    @property
    def duration(self):
        return self.end_time - self.start_time if self.end_time - self.start_time > 0 \
            else self._ticks_per_beat / UNIT

    @property
    def note_duration(self):
        return DURATION_TO_WORD[take_closest(DURATION_NOTES, self._ticks_per_beat / self.duration)]

    @property
    def velocity(self):
        return self._velocity

    @property
    def start_time(self):
        return self.__time_reposition(self._start_time)

    @property
    def start_beat(self):
        return int(self.start_time // (self._ticks_per_beat / UNIT))

    @property
    def end_time(self):
        return self.__time_reposition(self._end_time)

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def instrument(self):
        return self._instrument_type

    def __time_reposition(self, value):
        unit = self._ticks_per_beat / UNIT
        div = round(value / unit)
        return div * unit


class MidiTokenizer(object):
    def __init__(self):
        self.vocab = Vocabulary()
        self.stack = []

    def tokenize(self, midi_fp, save_txt_dir=None):
        """
        seperated by ticks_per_beat
        :return:
        """
        try:
            mid = MidiFile(midi_fp)
        except:
            return None

        current_time = 0
        notes = self.__get_all_notes(mid)
        tokens = []

        for idx, note in enumerate(notes):
            if note.is_drum:
                tokens.append("i0d")
            else:
                tokens.append("i" + str(note.instrument))
            tokens.append("n" + str(note.note))
            tokens.append(note.note_duration)
            tokens.append("v" + str(note.velocity))

            if idx == len(notes) - 1:  # last note
                tokens.append(END_TOKEN)
            else:
                next_note = notes[idx+1]
                if next_note.start_time > note.start_time:
                    sep_time = next_note.start_time - note.start_time
                    sep_duration = DURATION_TO_WORD[take_closest(DURATION_NOTES, note._ticks_per_beat / sep_time)]
                    tokens.append(SEP_TOKEN)
                    tokens.append(sep_duration)

        if save_txt_dir and tokens is not None:
            fname = str(os.path.basename(midi_fp).split('.')[0])
            with open(os.path.join(save_txt_dir, fname + '.txt'), 'w') as f:
                f.write('\n'.join(tokens))

        return tokens

    def stoi(self, tokens):
        return [self.vocab.word2idx[token] for token in tokens]

    def itos(self, ids):
        return [self.vocab.idx2word[id] for id in ids]

    def de_tokenize(self, tokens, save_midi_fp, tempo=DEFAULT_TEMPO, ticks_per_beat=DEFAULT_RESOLUTION):
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=ticks_per_beat)
        start = 0.
        end = 0.

        # First get how many tracks there are
        inst_token_types = []
        for token in tokens:
            if token.startswith('i') and token not in inst_token_types:
                inst_token_types.append(token)

        if len(inst_token_types) == 0:
            return False

        inst_dict = dict()
        for inst_token in inst_token_types:
            is_drum = False
            if inst_token[-1] == 'd':  # This instrument is a drum, need to be in channel 9
                program = 0
                is_drum = True
            else:
                program = int(inst_token[1:])

            inst_dict[inst_token] = pretty_midi.Instrument(program, is_drum)

        # Organize tokens
        organized_tokens = []
        skip_indexes = []
        for i, token in enumerate(tokens):
            if skip_indexes:
                skip_indexes.remove(i)
            else:
                current_group = []  # for note: [i, n, d, v]      for sep: [<sep>, d]
                if token in CTRL_TOKENS:
                    continue
                elif token == END_TOKEN:
                    break
                elif token == SEP_TOKEN:
                    if i == len(tokens) - 1:  # `token` is the last token
                        break
                    elif tokens[i + 1] in DURATION_TOKENS:
                        current_group.append(token)
                        current_group.append(tokens[i + 1])
                        skip_indexes.append(i + 1)
                        organized_tokens.append(current_group)
                elif token.startswith('i') and i <= len(tokens) - 4:
                    if tokens[i].startswith('i') and tokens[i + 1].startswith('n') and \
                            tokens[i + 2] in DURATION_TOKENS and tokens[i + 3].startswith('v'):
                        current_group.extend(tokens[i:i + 4])
                        skip_indexes.extend([i+1, i+2, i+3])
                        organized_tokens.append(current_group)

        if len(organized_tokens) == 0:
            return False

        # Build Each Note
        abs_time = 0  # start time (in seconds) of the current notes group
        for i, group in enumerate(organized_tokens):  # iterate each token group
            if group[0] == SEP_TOKEN:
                sep_time = (1.0 / WORD_TO_DURATION[group[1]]) / tempo * 60
                abs_time += sep_time
            else:
                inst_token = group[0]
                pitch = int(group[1][1:])
                duration = (1.0 / WORD_TO_DURATION[group[2]]) / tempo * 60
                velocity = int(group[3][1:])
                inst_dict[inst_token].notes.append(pretty_midi.Note(velocity, pitch, abs_time, abs_time + duration))
                if abs_time + duration > end:
                    end = abs_time + duration

        pm.instruments = list(inst_dict.values())

        # Create indicator for end of song
        eos = pretty_midi.TimeSignature(1, 1, end - start)
        pm.time_signature_changes.append(eos)

        # Save MIDI file
        pm.write(save_midi_fp)
        return True

    def __push_note(self, note):
        self.stack.append(note)

    def __pop_note(self, note):
        i = -1
        length = len(self.stack)
        for index, n in enumerate(reversed(self.stack)):
            if n.note == note:
                i = index
                break
        if i == -1:
            return None
        return self.stack.pop(length - i - 1)

    def __find_instrument_type(self, channel, mid):
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, "channel") and msg.channel != channel:
                    continue
                if msg.type == "program_change":
                    return msg.program
        return -1

    def __get_all_notes(self, mid):
        """
        Inside a track, it is delta time in ticks. This must be an integer.
        A beat is the same as a quarter note.
        Beats are divided into ticks, the smallest unit of time in MIDI.
        """
        ticks_per_beat = mid.ticks_per_beat  # all of lakh and mirex-like set default as 480
        notes = []

        for i, track in enumerate(mid.tracks):
            timer = 0
            instrument_type = -1
            notes_per_inst = []
            for msg in track:
                timer += msg.time
                if msg.type in CAPTURE_MSG:
                    if msg.type is "note_on" and msg.velocity != 0:
                        channel = msg.channel
                        if instrument_type == -1:
                            if channel != 9:
                                # 遍历其他 track 检查 instrument type
                                instrument_type = self.__find_instrument_type(channel, mid)
                                if instrument_type == -1:
                                    for default_ins in INSTRUMENT_FAMILY:
                                        if default_ins in track.name.lower():
                                            instrument_type = DEFUALT_INSTRUMENT[default_ins]
                            else:
                                instrument_type = 0
                        if instrument_type == -1:
                            instrument_type = 0
                        note = msg.note
                        velocity = msg.velocity
                        n = Note(note, velocity, timer, instrument_type, channel, ticks_per_beat)
                        # save the note to the stack
                        self.__push_note(n)

                    if msg.type is "note_off" or (msg.type is "note_on" and msg.velocity == 0):
                        # find the start note and fill the end time
                        n = self.__pop_note(msg.note)
                        if n is None:
                            continue
                        n.end_time = timer
                        # push note to the track list
                        notes_per_inst.append(n)

                    if msg.type is "program_change":
                        instrument_type = msg.program

            notes.extend(notes_per_inst)

        notes.sort(key=lambda x: (x.start_time, x.instrument))

        return notes


# %% Unit Test
if __name__ == "__main__":
    tokenizer = MidiTokenizer()

    vocab = tokenizer.vocab
    print("vocab size: ", len(vocab))
    vocab.save()

    midi_fp = "./plot_figures_in_paper/example.mid"

    save_txt_path = './plot_figures_in_paper'
    os.makedirs(save_txt_path, exist_ok=True)

    tokens = tokenizer.tokenize(midi_fp, save_txt_path)
    ids = tokenizer.stoi(tokens)
    de_tokens = tokenizer.itos(ids)
    tokenizer.de_tokenize(de_tokens, './plot_figures_in_paper/example_.mid')
