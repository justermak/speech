from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        windows = []
        padded_waveform = np.pad(waveform, (self.window_size // 2, self.window_size))
        for i in range(0, len(waveform) - self.window_size % 2 + 1, self.hop_length):
            windows.append(padded_waveform[i:i + self.window_size])
        
        return np.array(windows)
    

class Hann:
    def __init__(self, window_size=1024):
        self.window = scipy.signal.windows.hann(window_size, sym=False)

    
    def __call__(self, windows):
        return windows * self.window



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        spec = np.fft.rfft(windows)
        spec = np.abs(spec)
        if self.n_freqs:
            spec = spec[:, :self.n_freqs]
        return spec


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=1, fmax=8192).T
        self.inv = np.linalg.pinv(self.mel)

    def __call__(self, spec):
        mel = spec @ self.mel

        return mel

    def restore(self, mel):
        spec = mel @ self.inv

        return spec


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1]



class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor


    def __call__(self, mel):
        return mel * self.loudness_factor




class PitchUp:
    def __init__(self, num_mels_up):
        self.num_mels_up = num_mels_up


    def __call__(self, mel):
        return np.concatenate((np.zeros((mel.shape[0], self.num_mels_up)), mel[:, :-self.num_mels_up]), axis=1)



class PitchDown:
    def __init__(self, num_mels_down):
        self.num_mels_down = num_mels_down


    def __call__(self, mel):
        return np.concatenate((mel[:, self.num_mels_down:], np.zeros((mel.shape[0], self.num_mels_down))), axis=1)



class SpeedUpDown:
    EPS = 1e-8
    def __init__(self, speed_up_factor=1.0):
        self.speed_up_factor = speed_up_factor
    
    @staticmethod
    def my_round(x, ln):
        if x - int(x) < 0.5 - SpeedUpDown.EPS:
            return int(x)
        if x - int(x) > 0.5 + SpeedUpDown.EPS:
            return int(x) + 1
        if ln % 2 == 0:
            return int(x)
        return int(x) + 1

    def __call__(self, mel):
        res = np.zeros((int(self.speed_up_factor * mel.shape[0]), mel.shape[1]))
        sp = res.shape[0] / mel.shape[0]
        for i in range(0, res.shape[0]):
            res[i] = mel[SpeedUpDown.my_round(i / sp, res.shape[0])]
        return res



class FrequenciesSwap:
    def __call__(self, mel):
        return mel[:, ::-1]



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.quantile = quantile


    def __call__(self, mel):
        mel1 = mel.copy()
        q = np.quantile(mel.ravel(), self.quantile)
        mel1[mel1 < q] = 0
        return mel1



class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

