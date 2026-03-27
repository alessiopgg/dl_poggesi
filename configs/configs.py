"""
config.py — Configurazione centralizzata per il progetto MAE-AST.

Tutti gli iperparametri sono raccolti qui. Le fonti sono:
  - Paper MAE-AST (Baade et al., Interspeech 2022), Sezione 2.1-2.3
  - Repo MAE-AST: https://github.com/AlanBaade/MAE-AST-Public
  - Paper MAE vision (He et al., 2021)
  - Repo SSAST: https://github.com/YuanGongND/ssast
  - Nostri adattamenti per FSD50K / ESC-50

Uso:
    from configs.config import AudioConfig, PatchConfig, EncoderConfig, ...

    # Accesso ai parametri
    print(AudioConfig.SAMPLE_RATE)       # 16000
    print(EncoderConfig.NUM_LAYERS)      # 6

    # Per ablazioni, modifica direttamente i valori:
    EncoderConfig.NUM_LAYERS = 12
"""

from dataclasses import dataclass, field


# ======================================================================
# AUDIO & PREPROCESSING
# Paper Sezione 2.1: "we take as input a 16khz audio waveform and
# convert it into 128-dimensional log Mel filterbank features with
# a frame length of 25ms and frame shift of 10ms"
# ======================================================================

@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000            # Resample a 16kHz (paper Sez. 2.1)
    N_MELS: int = 128                   # Dimensione frequenza del mel spectrogram
    N_FFT: int = 400                    # Frame length = 25ms @ 16kHz (0.025 * 16000)
    HOP_LENGTH: int = 160               # Frame shift = 10ms @ 16kHz (0.010 * 16000)
    F_MIN: float = 20.0                 # Frequenza minima per il mel filterbank
    F_MAX: float = 8000.0               # Frequenza massima per il mel filterbank

    # Durata target audio (in secondi)
    # FSD50K ha clip variabili (0.3-550s), facciamo pad/truncate
    # AudioSet usa 10s, noi facciamo lo stesso per FSD50K
    PRETRAIN_AUDIO_LENGTH_SEC: float = 10.0     # Pretraining (FSD50K)
    FINETUNE_AUDIO_LENGTH_SEC: float = 5.0      # Fine-tuning (ESC-50, clip da 5s)

    # Normalizzazione
    # Paper: "normalize to mean 0 and standard deviation 1/2"
    # Opzione A (paper): per-sample normalization
    # Opzione B (SSAST repo): statistiche globali del dataset
    # Noi: per-sample come default, calcoleremo anche globali su FSD50K
    NORM_MEAN: float = None             # None = per-sample, altrimenti float
    NORM_STD: float = None              # None = per-sample, altrimenti float
    TARGET_STD: float = 0.5             # Paper: "standard deviation 1/2"


# ======================================================================
# PATCHING (Tokenizzazione)
# Paper Sez. 2.1 e Figura 1:
# Patch-based: 16 filter × 16 frame → 256 valori per patch
# Frame-based: 128 filter × 2 frame → 256 valori per frame
# "patch input being unfolded into a one-dimensional stream
#  ordered first by channel and second by time"
# No overlap durante pretraining e fine-tuning (a differenza di AST)
# ======================================================================

@dataclass
class PatchConfig:
    # Patch-based (default, migliore per audio tasks)
    PATCH_H: int = 16                   # Altezza patch (frequenza)
    PATCH_W: int = 16                   # Larghezza patch (tempo)
    PATCH_STRIDE_H: int = 16            # Stride frequenza (no overlap)
    PATCH_STRIDE_W: int = 16            # Stride tempo (no overlap)

    # Frame-based (alternativa, migliore per speech tasks)
    # Per usare frame-based: PATCH_H=128, PATCH_W=2, strides uguali
    # FRAME_H: int = 128
    # FRAME_W: int = 2

    @property
    def PATCH_DIM(self) -> int:
        """Dimensione del vettore per ogni patch: 16*16 = 256."""
        return self.PATCH_H * self.PATCH_W

    def num_patches(self, n_mels: int, time_frames: int) -> int:
        """
        Calcola il numero totale di patch dallo spettrogramma.

        Per 10s con patch 16x16: (128/16) * (1001/16) = 8 * 62 = 496
        Per 5s con patch 16x16:  (128/16) * (501/16)  = 8 * 31 = 248
        """
        n_freq = n_mels // self.PATCH_STRIDE_H
        n_time = time_frames // self.PATCH_STRIDE_W
        return n_freq * n_time


# ======================================================================
# ENCODER (ViT standard, solo su token NON mascherati)
# Paper Sez. 2.1: "By default, we use an encoder with 6 layers"
# "Both the encoder and decoder use 12 heads and a width of 768"
# Repo: model.encoder_layers=12 per i risultati migliori (Table 1)
# ======================================================================

@dataclass
class EncoderConfig:
    NUM_LAYERS: int = 6                 # Default paper; 12 per risultati migliori
    EMBED_DIM: int = 768                # Dimensione embedding (width)
    NUM_HEADS: int = 12                 # Multi-head attention heads
    MLP_RATIO: float = 4.0             # FFN dim = EMBED_DIM * MLP_RATIO = 3072
    DROPOUT: float = 0.0               # Dropout rate
    ATTENTION_DROPOUT: float = 0.0     # Dropout in attention


# ======================================================================
# DECODER (shallow, usato SOLO durante pretraining)
# Paper Sez. 2.1: "a decoder of 2 layers"
# "Both the encoder and decoder use 12 heads and a width of 768"
# NOTA: a differenza di MAE vision (decoder 512-d), qui è 768-d
# Table 2: decoder depth ablation mostra 2 layer ottimale
# ======================================================================

@dataclass
class DecoderConfig:
    NUM_LAYERS: int = 2                 # Paper default; Table 2 ablation
    EMBED_DIM: int = 768                # Stessa dim dell'encoder (diverso da MAE vision)
    NUM_HEADS: int = 12                 # Stessi heads dell'encoder
    MLP_RATIO: float = 4.0             # FFN dim = 768 * 4 = 3072
    DROPOUT: float = 0.0


# ======================================================================
# MASKING
# Paper Sez. 2.2: "we mask by shuffling the input patches and
# keeping the first 1-p proportion of tokens"
# Table 3: 75% chunk masking → migliori risultati patch-based
# Repo: model.random_mask_prob=0.75 task.mask_type="chunk_mask"
# ======================================================================

@dataclass
class MaskConfig:
    MASK_RATIO: float = 0.75            # 75% mascherato (paper default)
    MASK_STRATEGY: str = "random"       # "random", "chunk", "span"

    # Chunk masking params (SSAST-style, per patch-based)
    # Paper: "selects C ∈ {3,4,5} and randomly masks C×C chunks"
    CHUNK_SIZES: list = field(default_factory=lambda: [3, 4, 5])

    # Span masking params (Wav2Vec2-style, per frame-based)
    # Paper: "spans of length M=10"
    SPAN_LENGTH: int = 10


# ======================================================================
# LOSS
# Paper Sez. 2.3:
# - Reconstruction: MSE tra output decoder e patch mascherato originale
# - Classification: InfoNCE, negativi dalla stessa clip audio
# - Combinata: classification_loss + lambda * reconstruction_loss
# - lambda = 10 (uguale a SSAST)
# Repo: criterion.classification_weight=1 criterion.reconstruction_weight=10
# ======================================================================

@dataclass
class LossConfig:
    CLASSIFICATION_WEIGHT: float = 1.0      # Peso loss discriminativa (InfoNCE)
    RECONSTRUCTION_WEIGHT: float = 10.0     # Peso loss generativa (MSE) = lambda
    INFONCE_TEMPERATURE: float = 0.1        # Temperatura per InfoNCE


# ======================================================================
# PRETRAINING
# Paper Sez. 2.1: "Adam optimizer with weight decay 0.01,
# initial learning rate 0.0001, and a polynomial decay learning
# rate scheduler... batch size of 32... up to 600,000 iterations"
# Figure 2: converge in 8 epoche (~500k iterazioni su AudioSet)
#
# Per noi (FSD50K, ~41k clip):
#   Con batch_size=32: ~1281 step/epoca
#   8 epoche → ~10k step
# ======================================================================

@dataclass
class PretrainConfig:
    # Optimizer
    OPTIMIZER: str = "adam"
    LEARNING_RATE: float = 1e-4         # 0.0001 (paper)
    WEIGHT_DECAY: float = 0.01          # Paper Sez. 2.1
    BETAS: tuple = (0.9, 0.999)         # Adam default

    # Scheduler
    LR_SCHEDULER: str = "polynomial"    # Polynomial decay (paper)
    WARMUP_STEPS: int = 500             # Warmup iniziale

    # Training
    BATCH_SIZE: int = 32                # Paper: "roughly equivalent to batch size of 32"
    MAX_EPOCHS: int = 8                 # Paper Figure 2: converge in 8 epoche
    MAX_STEPS: int = None               # Se impostato, sovrascrive MAX_EPOCHS

    # Logging e checkpoint
    LOG_INTERVAL: int = 200             # Repo: common.log_interval=200
    SAVE_INTERVAL_STEPS: int = 2500     # Salva checkpoint ogni N step


# ======================================================================
# FINE-TUNING (su ESC-50)
# Paper: "we use the fine-tuning pipeline from SSAST"
# Solo encoder (decoder scartato), mean pooling → classification head
# 5-fold cross-validation su ESC-50
# ======================================================================

@dataclass
class FinetuneConfig:
    # Optimizer
    LEARNING_RATE: float = 1e-4         # Da SSAST, o grid search
    WEIGHT_DECAY: float = 0.01
    BETAS: tuple = (0.9, 0.999)

    # Training
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 50                # Fine-tuning epochs

    # ESC-50 specifico
    NUM_CLASSES: int = 50               # ESC-50 ha 50 classi
    NUM_FOLDS: int = 5                  # 5-fold cross-validation


# ======================================================================
# PATH (configurabili per ogni ambiente)
# ======================================================================

@dataclass
class PathConfig:
    # Dataset
    FSD50K_DIR: str = "D:/deep_learning"
    ESC50_DIR: str = "D:/deep_learning/ESC-50-master"

    # Datafiles JSON (generati dagli script prepare_*)
    DATAFILES_DIR: str = "./datafiles"

    # Output
    CHECKPOINT_DIR: str = "./checkpoints"
    LOG_DIR: str = "./logs"