# get the VITS model and change the train.py 

import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig , CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.downloaders import download_thorsten_de
# choose the right formatter, provide the the meta file that include text of audios x.csv, path to audios
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="ljspeech2", meta_file_train=" ", path=" " 
)

# chagne sample rate according to your sample rate 
audio_config = VitsAudioConfig(
    sample_rate=48000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

characters_config=CharactersConfig(
characters_class="TTS.tts.models.vits.VitsCharacters",
characters="\u0650\u0629\u0649\u0623\u0628\u062A\u062B\u062C\u062D\u062E\u0640\u0621\u0627\u0626\u06A9\u06CC\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u064A\u064B\u064C\u064D\u064E\u064F\u065D\u0651\u0652\u0622\u0624\u0625",
punctuations= "!()=,-.:;؟+<>\u060C\u061B\u0020\u005F\u003F\u0022\u00AB\u00BB",
pad="<PAD>",
eos="<EOS>",
bos="<BOS>",
blank="<BLNK>",
)
config = VitsConfig(
    dashboard_logger='tensorboard',
    audio=audio_config,
    run_name="", # provide a name for your model
    batch_size=8, # this can be 16 if your system allow more than 8 batch size.. 
    eval_batch_size=8,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    save_step=1000,
    text_cleaner="basic_cleaners", #"collapse_whitespace"
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="ar",
    compute_input_seq_cache=True,
    print_step=50,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters = characters_config,
    print_eval=True,
    cudnn_benchmark=False,
    test_sentences = [
         [

            "خدم",

        ],

        [

            "ودّم وصدّع",      

        ],

        [

            "هدّم",

        ]
    ]
)


# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)


# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
