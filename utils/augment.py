import numpy as np
from collections import defaultdict

from data_config import TOKENIZATION, PITCH_MIN, PITCH_MAX, VELOCITY_MIN, VELOCITY_MAX


def instrument_augment(tokens):
    # change the instrument in the same instrument family, i.e., 0 Acoustic Grand Piano -> 1 Bright Acoustic Piano
    tokens_augmented = []

    if TOKENIZATION == 'time':
        # First get how many tracks there are
        tracks_name = []
        for word in tokens:
            if word.startswith('i') and word not in tracks_name:
                tracks_name.append(word)

        old_to_new_tracks_name = dict()
        for track_name in tracks_name:
            if track_name[-1] == 'd':
                old_to_new_tracks_name[track_name] = track_name  # don't need to change drums
            else:
                instrument_type = int(track_name[1:])
                if 0 <= instrument_type <= 7:  # piano family
                    old_to_new_tracks_name[track_name] = 'i' + str(np.random.randint(0, 8))
                elif 16 <= instrument_type <= 23:  # organ family
                    old_to_new_tracks_name[track_name] = 'i' + str(np.random.randint(16, 24))
                elif 24 <= instrument_type <= 31:  # guitar family
                    old_to_new_tracks_name[track_name] = 'i' + str(np.random.randint(24, 32))
                elif 32 <= instrument_type <= 39:  # bass family
                    old_to_new_tracks_name[track_name] = 'i' + str(np.random.randint(32, 40))
                elif 40 <= instrument_type <= 47:  # strings family
                    old_to_new_tracks_name[track_name] = 'i' + str(np.random.randint(40, 48))
                else:
                    old_to_new_tracks_name[track_name] = track_name  # don't change other instruments

        for token in tokens:
            if token.startswith('i'):
                tokens_augmented.append(old_to_new_tracks_name[token])
            else:
                tokens_augmented.append(token)

    elif TOKENIZATION == 'track':
        # TODO
        print("TODO: instrument_augment - track")
    return tokens_augmented


def pitch_transpose(tokens, transpose_amt=0):
    if transpose_amt == 0:
        return tokens

    tokens_transposed = []

    if TOKENIZATION == 'time':
        for token in tokens:
            if token[0] == 'n':
                midi_note = int(token[1:])
                new_midi_note = midi_note + transpose_amt
                if PITCH_MIN <= new_midi_note <= PITCH_MAX:
                    tokens_transposed.append('n{}'.format(new_midi_note))
                elif new_midi_note < PITCH_MIN:
                    tokens_transposed.append('n{}'.format(PITCH_MIN))
                elif new_midi_note > PITCH_MAX:
                    tokens_transposed.append('n{}'.format(PITCH_MAX))
            else:
                tokens_transposed.append(token)
    elif TOKENIZATION == 'track':
        for token in tokens:
            if token[0] == '<':
                tokens_transposed.append(token)
                continue

            event_type, event_value = token.split(':')

            if event_type == 'note_on':
                new_event_value = int(event_value) + transpose_amt
                if PITCH_MIN <= new_event_value <= PITCH_MAX:
                    tokens_transposed.append('note_on:{}'.format(new_event_value))
                # TODO: current note is abandoned, so delete related instrument, duration, and velocity
            else:
                tokens_transposed.append(token)

    return tokens_transposed


def time_stretch(tokens, playback_speed=1.):

    if TOKENIZATION == 'time':
        # tokenization_by_time的编码方式不用做time stretch，因为最小时间步是32分音符，可设置tempo来控制绝对时间
        return tokens

    elif TOKENIZATION == 'track':
        # TODO
        print("TODO: time_stretch - track")
        return tokens


def velocity_augment(tokens, augment_value=0):
    if augment_value == 0:
        return tokens

    tokens_augmented = []

    if TOKENIZATION == 'time':
        for token in tokens:
            if token[0] == 'v':
                velocity = int(token[1:])
                new_velocity = velocity + augment_value
                if VELOCITY_MIN <= new_velocity <= VELOCITY_MAX:
                    tokens_augmented.append('v{}'.format(new_velocity))
                elif new_velocity < VELOCITY_MIN:
                    tokens_augmented.append('v{}'.format(VELOCITY_MIN))
                elif new_velocity > VELOCITY_MAX:
                    tokens_augmented.append('v{}'.format(VELOCITY_MAX))
            else:
                tokens_augmented.append(token)
    elif TOKENIZATION == 'track':
        # TODO:
        print("TODO: velocity_augment - track")
    return tokens_augmented


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_txt_fp', type=str)
    parser.add_argument('output_txt_fp', type=str)
    parser.add_argument('--transpose_amt', type=int)
    parser.add_argument('--playback_speed', type=float)

    parser.set_defaults(transpose_amt=0, playback_speed=1.)

    args = parser.parse_args()

    with open(args.input_txt_fp, 'r') as f:
        events = f.read().strip().splitlines()

    events = pitch_transpose(events, args.transpose_amt)
    events = time_stretch(events, args.playback_speed)

    with open(args.output_txt_fp, 'w') as f:
        f.write('\n'.join(events))
