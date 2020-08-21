# for data_cleaning
MIN_NUM_INSTRUMENTS = 1
FILTER_MID_LEN_BELOW_SECONDS = 1.
FILTER_MID_LEN_ABOVE_SECONDS = 1200.
FILTER_MID_BAD_TIMES = True
PITCH_MIN = 21
PITCH_MAX = 108
VELOCITY_MIN = 21
VELOCITY_MAX = 108
QUANTIZE_VELOCITY = True  # 为True时，VELOCITY_MIN必须设为20以上，VELOCITY_MAX必须设为110以下
FILTER_INS_DUPLICATE = True
OUTPUT_MAX_NUM_SECONDS = 1200.
DEFAULT_RESOLUTION = 220   # 对应mido里的ticks_per_beat
DEFAULT_TEMPO = 120


# for tokenization
DEFAULT_SAVING_PROGRAM = 0

INST_ENSEMBLE_SIZE = 128
if INST_ENSEMBLE_SIZE == 5:
    DEFAULT_LOADING_PROGRAMS = [-1, 0, 24, 32, 48]
    # PROGRAM_INST_MAP = {-1: 'drums', 0: 'piano', 24: 'guitar', 32: 'bass', 48: 'strings'}
elif INST_ENSEMBLE_SIZE == 17:
    DEFAULT_LOADING_PROGRAMS = [-1, 0, 24, 32, 48]
    # PROGRAM_INST_MAP = {-1: 'drums', 0: 'piano', 24: 'guitar', 32: 'bass', 48: 'strings'}
elif INST_ENSEMBLE_SIZE == 128:
    DEFAULT_LOADING_PROGRAMS = range(-1, 128)
    # programs = [i for i in DEFAULT_LOADING_PROGRAMS]
    # inst_tokens = [f'instrument{i}' for i in DEFAULT_LOADING_PROGRAMS]  # instrument-1 is drum
    # PROGRAM_INST_MAP = dict(zip(programs, inst_tokens))

DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(PITCH_MIN, PITCH_MAX + 1)
DEFAULT_VELOCITY_RANGE = range(VELOCITY_MIN, VELOCITY_MAX + 1)
# DEFAULT_NORMALIZATION_BASELINE = 60  # C4

BEAT_LENGTH = 60 / DEFAULT_TEMPO                           # seconds per beat
DEFAULT_VELOCITY_STEPS = 32
DEFAULT_TIME_SHIFT_STEPS = 256
DEFAULT_NOTE_LENGTH = BEAT_LENGTH


CTRL_TOKENS = ['<animated_cartoon>', '<lovely>', '<hk_tw>', '<classical>',
               '<jazz>', '<usa_euro_pop>', '<christmas>', '<games>']

