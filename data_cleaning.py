"""
1. Basic data cleaning:

- Ignore unusually large MIDI files (controlled by: MAX_FILE_SIZE)
- Filter out MIDIs with extreme short or long length (controlled by: FILTER_MID_LEN_BELOW_SECONDS, FILTER_MID_LEN_ABOVE_SECONDS)
- Filter out negative times (controlled by: FILTER_MID_BAD_TIMES)
#- Quantize to audio samples
- Filter out instruments with bizarre pitch ranges
- Filter out duplicate instruments


2. Narrowing down to a fixed ensemble (5 or 17)

- 5 Basic band instruments: {Drums, Piano, Guitar, Bass, Strings}
    like LPD-5:
        https://salu133445.github.io/lakh-pianoroll-dataset/dataset
    Concretely,
        Instruments are merged into five common categories: Drums, Piano, Guitar, Bass and Strings according to the
        program numbers provided in the MIDI files. Note that instruments out of the five categories are considered
        as part of the strings except those in the Percussive, Sound effects and Synth Effects families (those are
        abandoned).
            Drums: pretty_midi.Instrument.is_drum=True               -> 0 (Percussion program)
            Piano: 0 <= pretty_midi.Instrument.program <= 7          -> 0    'Acoustic Grand Piano'
            Guitar: 24 <= ~ <= 31                                    -> 24   'Acoustic Guitar (nylon)'
            Bass: 32 <= ~ <= 39                                      -> 32   'Acoustic Bass'
            Strings: 40 ~ 47, conbined with 8~23, 48~95, 104~111     -> 48   'String Ensemble 1'


TODO: 17 instrument families
- 17 categories (drums and 16 instrument families):
{Drums, Piano, Chromatic Percussion, Organ, Guitar, Bass, Strings, Ensemble, Brass, Reed, Pipe, Synth Lead, Synth Pad,
Synth Effects, Ethnic, Percussive, Sound Effects}
    like LPD-17:
        https://salu133445.github.io/lakh-pianoroll-dataset/dataset
    Concretely,
        Refer to: https://www.midi.org/specifications/item/gm-level-1-sound-set

"""

import os

import pretty_midi

# Default Parameters
from data_config import *


# pretty_midi.pretty_midi.MAX_TICK = 1e16


def quantize_v(v):
    return int(v/10)*10 + 5


def save_as_cleaned_midi(raw_midi_fp, dataset_name, output_dir, fix_ensemble_size=5, quantize_ticks=False):
    assert fix_ensemble_size == 5 or fix_ensemble_size == 17 or fix_ensemble_size == 128

    if dataset_name == 'lakh':
        midi_name = str(os.path.split(raw_midi_fp)[0].split('/')[-1]) + '-' + \
                    str(os.path.split(raw_midi_fp)[1].split('.')[0])
    else:
        midi_name = str(os.path.split(raw_midi_fp)[1].split('.')[0])

    if MIN_NUM_INSTRUMENTS <= 0:
        raise ValueError()

    ### Do filtering on the whole MIDI file level
    try:
        midi = pretty_midi.PrettyMIDI(raw_midi_fp)
    except:
        return

    # Filter MIDIs with extreme length
    midi_len = midi.get_end_time()
    if midi_len < FILTER_MID_LEN_BELOW_SECONDS or midi_len > FILTER_MID_LEN_ABOVE_SECONDS:
        return

    ### Do filtering on the Instrument and Note level
    instruments = midi.instruments

    # if use quantization, do as LakhNES does
    ticks_per_second = 44100 if quantize_ticks else None
    default_resolution = 22050 if quantize_ticks else DEFAULT_RESOLUTION

    # Filter out negative times  ## and Quantize to audio samples
    for ins in instruments:
        ins.remove_invalid_notes()
        for n in ins.notes:
            if FILTER_MID_BAD_TIMES:
                if n.start < 0 or n.end < 0 or n.end < n.start:
                    return
            if quantize_ticks:
                n.start = round(n.start * ticks_per_second) / ticks_per_second
                n.end = round(n.end * ticks_per_second) / ticks_per_second

    # Filter out instruments with bizarre pitch ranges
    instruments_normal_range = []
    for ins in instruments:
        pitches = [n.pitch for n in ins.notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        if min_pitch >= PITCH_MIN and max_pitch <= PITCH_MAX:
            instruments_normal_range.append(ins)
    instruments = instruments_normal_range
    if len(instruments) < MIN_NUM_INSTRUMENTS:
        return

    # Sort notes for proper saving
    for ins in instruments:
        ins.notes = sorted(ins.notes, key=lambda x: x.start)

    # Filter out duplicate instruments
    if FILTER_INS_DUPLICATE:
        uniques = set()
        instruments_unique = []
        for ins in instruments:
            pitches = ','.join(['{}:{:.1f}'.format(str(n.pitch), n.start) for n in ins.notes])
            if pitches not in uniques:
                instruments_unique.append(ins)
                uniques.add(pitches)
        instruments = instruments_unique
        if len(instruments) < MIN_NUM_INSTRUMENTS:
            return

    # TODO: Find instruments that have a substantial fraction of the number of total notes

    # TODO: ensure tempo and other metadata is alright

    # TODO: ensure number of notes is alright

    ### Create assignments of MIDI instruments to fixed-ensemble instruments
    pm = pretty_midi.PrettyMIDI(initial_tempo=DEFAULT_TEMPO, resolution=default_resolution)
    start = None
    end = None

    if fix_ensemble_size == 5:

        mid_ins_notes = [[], [], [], [], []]  # drums, piano, guitar, bass, strings
        for ins in instruments:
            # similar as: https://github.com/salu133445/lakh-pianoroll-dataset/blob/master/src/merger_5.py
            if ins.is_drum:
                mid_ins_notes[0].extend(ins.notes)
            elif 0 <= ins.program <= 7:  # Piano
                mid_ins_notes[1].extend(ins.notes)
            elif 8 <= ins.program <= 23:  # Strings (lpd数据集的处理没有要这部分，需斟酌)
                mid_ins_notes[4].extend(ins.notes)
            elif 24 <= ins.program <= 31:  # Guitar
                mid_ins_notes[2].extend(ins.notes)
            elif 32 <= ins.program <= 39:  # Bass
                mid_ins_notes[3].extend(ins.notes)
            elif 40 <= ins.program <= 95 or 104 <= ins.program <= 111:  # Strings
                mid_ins_notes[4].extend(ins.notes)

        # TODO: Remove duplicate notes

        # Calculate length of this ensemble,
        # and find time of the very beginning note (start) and ending note (end) among all instruments
        for notes in mid_ins_notes:
            if notes is None or len(notes) == 0:
                continue
            notes = sorted(notes, key=lambda x: x.start)

            ins_start = min([n.start for n in notes])
            ins_end = max([n.end for n in notes])
            if start is None or ins_start < start:
                start = ins_start
            if end is None or ins_end > end:
                end = ins_end

        # Clip if needed
        if (end - start) > OUTPUT_MAX_NUM_SECONDS:
            end = start + OUTPUT_MAX_NUM_SECONDS

        # Create MIDI instruments
        drums_i = pretty_midi.Instrument(program=0, name='Drums', is_drum=True)
        piano_i = pretty_midi.Instrument(program=0, name='Piano', is_drum=False)
        guitar_i = pretty_midi.Instrument(program=24, name='Guitar', is_drum=False)
        bass_i = pretty_midi.Instrument(program=32, name='Bass', is_drum=False)
        strings_i = pretty_midi.Instrument(program=48, name='Strings', is_drum=False)

        # Create notes
        for ins_notes, ins_name, ins in zip(mid_ins_notes, ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings'],
                                            [drums_i, piano_i, guitar_i, bass_i, strings_i]):
            if ins_notes is None:
                continue

            for n in ins_notes:
                nvelocity = n.velocity
                npitch = n.pitch
                nstart = n.start
                nend = n.end

                nvelocity = max(nvelocity, VELOCITY_MIN)
                nvelocity = min(nvelocity, VELOCITY_MAX)
                if QUANTIZE_VELOCITY:
                    nvelocity = quantize_v(nvelocity)

                assert nstart >= start
                if nend > end:
                    continue
                assert nend <= end

                # The next two lines are trimming the blank in the beginning of MIDI file
                nstart = nstart - start
                nend = nend - start

                ins.notes.append(pretty_midi.Note(nvelocity, npitch, nstart, nend))

        # Add instruments to MIDI file
        pm.instruments.extend([drums_i, piano_i, guitar_i, bass_i, strings_i])

    elif fix_ensemble_size == 17:
        print("Ensemble 17 is to be implemented.")

    elif fix_ensemble_size == 128:

        # Calculate length of this ensemble,
        # and find time of the very beginning note and ending note among all instruments
        for ins in instruments:
            notes = ins.notes
            if notes is None or len(notes) == 0:
                continue
            notes = sorted(notes, key=lambda x: x.start)

            ins_start = min([n.start for n in notes])
            ins_end = max([n.end for n in notes])
            if start is None or ins_start < start:
                start = ins_start
            if end is None or ins_end > end:
                end = ins_end

        # Clip if needed
        if (end - start) > OUTPUT_MAX_NUM_SECONDS:
            end = start + OUTPUT_MAX_NUM_SECONDS

        # Create notes
        for ins in instruments:
            new_ins = pretty_midi.Instrument(program=ins.program, name=ins.name, is_drum=ins.is_drum)

            ins_notes = ins.notes

            if ins_notes is None:
                continue

            for n in ins_notes:
                nvelocity = n.velocity
                npitch = n.pitch
                nstart = n.start
                nend = n.end

                nvelocity = max(nvelocity, VELOCITY_MIN)
                nvelocity = min(nvelocity, VELOCITY_MAX)
                if QUANTIZE_VELOCITY:
                    nvelocity = quantize_v(nvelocity)

                assert nstart >= start
                if nend > end:
                    continue
                assert nend <= end

                # The next two lines are trimming the blank in the beginning of MIDI file
                nstart = nstart - start
                nend = nend - start

                new_ins.notes.append(pretty_midi.Note(nvelocity, npitch, nstart, nend))
            pm.instruments.append(new_ins)

    # Create indicator for end of song
    eos = pretty_midi.TimeSignature(1, 1, end - start)
    pm.time_signature_changes.append(eos)

    # Save MIDI file
    out_fp = '{}.mid'.format(midi_name)
    out_fp = os.path.join(output_dir, out_fp)
    pm.write(out_fp)


if __name__ == '__main__':
    import sys
    import glob
    import shutil
    import multiprocessing

    from tqdm import tqdm

    dataset_name = sys.argv[1]

    if dataset_name == 'lakh':
        midi_fps = glob.glob('data/{}/midi/*/*.mid*'.format(dataset_name))
    else:
        midi_fps = glob.glob('data/{}/midi/*.mid*'.format(dataset_name))
    out_dir = './data/{}/cleaned_midi'.format(dataset_name)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


    def _task(x):
        save_as_cleaned_midi(x, dataset_name, out_dir, INST_ENSEMBLE_SIZE)


    with multiprocessing.Pool(8) as p:
        r = list(tqdm(p.imap(_task, midi_fps), total=len(midi_fps)))
