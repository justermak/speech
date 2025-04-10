from collections import defaultdict
from typing import Tuple

import torch
import soundfile as sf

from sources.fastpitch.common.plotting_utils import plot_spectrogram_to_numpy
from sources.fastpitch.common.whisper import WhisperModel
from sources.fastpitch.data import MelSpectrogram, TextMelCollate, Wav
from sources.fastpitch.model import FastPitch


@torch.no_grad()
def validate(
    model,
    criterion,
    val_loader,
    device,
    use_gt_durations=False,
    use_gt_pitch=False,
):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        val_meta = defaultdict(float)

        for batch in val_loader:
            batch = batch.to(device)
            y_pred = model(batch, use_gt_durations=use_gt_durations, use_gt_pitch=use_gt_pitch)
            _, meta = criterion(
                y_pred,
                (batch.mels, batch.durations, batch.text_lengths, batch.pitches),
                is_training=False,
                meta_agg="sum"
            )
            for k, v in meta.items():
                val_meta[k] += v

        val_meta = {k: v / len(val_loader.dataset) for k, v in val_meta.items()}
        val_loss = val_meta["loss"]

    if was_training:
        model.train()
    return val_loss.item(), val_meta


def get_gt_data(valset, max_val_audios: int) -> Tuple[dict, dict, dict]:
    examples = [row for i, row in enumerate(valset) if i < max_val_audios]
    gt_texts_dict = {
        f'gt_text/{i}': row['meta']['normalized_text'] for i, row in enumerate(examples)
    }
    gt_mels_dict = {
        f'gt_mel/{i}': plot_spectrogram_to_numpy(row['mel']) for i, row in enumerate(examples)
    }
    gt_audios_dict = {
        f'gt_audio/{i}': Wav(*sf.read(row['meta']['audio_path']))
        for i, row in enumerate(examples)
    }
    return gt_texts_dict, gt_mels_dict, gt_audios_dict


def generate_predictions(valset, fp: FastPitch, hfg,
                         max_predictions: int, device: str) -> Tuple[dict, dict, dict]:
    device = fp.device
    test_loader = torch.utils.data.DataLoader(valset, shuffle=False, collate_fn=TextMelCollate(), batch_size=max_predictions)
    batch = next(iter(test_loader))
      
    with torch.no_grad():
        mels, mel_lens, *_ = fp.infer(batch.to(device))
        mels = mels.permute(0, 2, 1)
        audios = hfg(mels)

    mel_lens = mel_lens.detach().cpu().numpy()
    audios = audios.detach().cpu().numpy()
    audio_lengths = [int(MelSpectrogram.frames_to_time(length) * MelSpectrogram.sample_rate) for length in mel_lens]
    audios = [
        Wav(audio[..., :length].squeeze(), sr=MelSpectrogram.sample_rate)
        for audio, length in zip(audios, audio_lengths)
    ]

    asr = WhisperModel(device=device)
    transcriptions = asr.predict(audios)

    mels = mels.detach().cpu().numpy()

    asr_texts_dict = {
        f'asr_output/{i}': transcriptions[i] for i in range(len(audios))
    }
    pred_mel_dict = {
        f'pred_mel/{i}': plot_spectrogram_to_numpy(mels[i]) for i in range(len(audios))
    }
    pred_audios_dict = {
        f'pred_audio/{i}': audios[i] for i in range(len(audios))
    }
    return asr_texts_dict, pred_mel_dict, pred_audios_dict