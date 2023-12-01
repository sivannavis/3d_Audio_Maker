"""
This script perform speaker reproduction of binaural audio using crosstalk cancellation.

The implementation follows:
Choueiri, Edgar Y. "Optimal crosstalk cancellation for binaural audio with two loudspeakers." Princeton University 28 (2008).
Roginska, Agnieszka, and Paul Geluso, eds. Immersive sound: the art and science of binaural and multi-channel audio. Taylor & Francis, 2017. Chapter 5.

Sivan Ding
sivan.d@nyu.edu
"""
import librosa
import numpy as np
import pyfar as pf
import soundfile as sf

SPEED = 340  # sound speed at 27∘C


class CrosstalkCancel:
    def __init__(self, sig, sr, geometry):
        self.sig = sig
        self.sr = sr
        self.l1 = geometry['l1']
        self.l2 = geometry['l2']
        self.delta_l = geometry['delta_l']
        self.delta_t = geometry['delta_t']
        self.theta = geometry['theta']
        self.g = self.l1 / self.l2
        self.max = np.amax(self.sig)
        self.counter = 0

        # configuration for preprocessing
        self.window = 128
        self.stride = 128

    def fft(self, signal):
        return np.fft.fft(signal)

    def xtc_filter(self, mono_signal, contra_hrtf, ipsi_hrtf, output):
        """

        :param mono_signal:
        :param contra_hrtf: far from this side of monosignal
        :param ipsi_hrtf: close to this side of monosignal
        :param output:
        :return:
        """
        attenuation = self.g
        delay = self.delta_t
        self.counter += 1

        # create recursion base condition and normal condition
        cancelled = self.freq_delay(mono_signal, delay)
        cancelled *= attenuation
        if self.counter % 2:
            cancelled *= contra_hrtf * freq_invert(ipsi_hrtf)
            output[1] += cancelled
        else:
            cancelled *= ipsi_hrtf * freq_invert(contra_hrtf)
            output[0] += cancelled

        if amp_to_db(np.amax(np.abs(cancelled)) / self.max) >= -60 or self.counter < 100:
            cancelled, _ = self.xtc_filter(cancelled, contra_hrtf, ipsi_hrtf, output)

        return cancelled, output

    def xtc_process(self, hrir):
        sig_fft = self.fft(self.sig)
        # pad hrtf to be same length as audio signal
        hrir = np.pad(hrir, ((0, 0), (0, self.sig.shape[1] - hrir.shape[1])), 'constant')
        hrtf = self.fft(hrir)  # 4 * length: l2l, l2r, r2l, r2r

        # core function of cancellation
        _, xtc_left = self.xtc_filter(self.sig[0], hrtf[2], hrtf[0], np.zeros_like(self.sig, dtype=complex))
        _, xtc_right = self.xtc_filter(self.sig[1], hrtf[1], hrtf[3], np.zeros_like(self.sig, dtype=complex))

        # do convolution for ipsilateral side
        sig_hrtf = np.zeros_like(sig_fft)
        sig_hrtf[0] = sig_fft[0] * hrtf[0]
        sig_hrtf[1] = sig_fft[1] * hrtf[3]

        # do inverse fft for all of them
        sig_ifft = np.real(np.fft.ifft(sig_hrtf))
        xtc_left_ifft = np.real(np.fft.ifft(xtc_left))
        xtc_right_ifft = np.real(np.fft.ifft(xtc_right))

        # sum left and right signals
        left = sig_ifft[0] + xtc_left_ifft[0] + xtc_right_ifft[1]  # sum up everything that goes to left
        right = sig_ifft[1] + xtc_left_ifft[1] + xtc_right_ifft[0]

        # normallization by channel
        left = left / max(left)
        right = right / max(right)

        stereo = np.vstack((left, right))

        print(self.counter)

        return stereo

    def freq_delay(self, mono_signal, second):
        l = mono_signal.size
        delay_sample = int(self.sr * second)
        delay = np.zeros_like(mono_signal, dtype=complex)

        for i in range(l):
            delay[i] += mono_signal[i] * np.exp(-1j * 2 * np.pi * i / l * delay_sample)

        return delay


def freq_invert(x):

    return np.true_divide(1, x)


def amp_to_db(x):
    return 20 * np.log10(x)


def get_ir(source_coordinates, data_ir):
    """
            all measurements are done at distance 2.06m with (azimuth, elevation, radius)
        :param source_coordinates:
        :param data_ir:
        :return: a list of l2l, l2r, r2l, r2r HRIR
        """
    location = ['left', 'right']
    azimuth = [30, 330]
    elevation = [0, 0]
    hrir = []
    for i, loc in enumerate(location):
        index = source_coordinates.find_nearest_k(
            azimuth[i], elevation[i], 2.06, k=1, domain='sph', convention='top_elev', unit='deg')[
            0]  # hardcoded 2.06 m radius
        print(loc, index, source_coordinates.cartesian[index])
        hrir.append(data_ir[index].time[0])
        hrir.append(data_ir[index].time[1])

    return np.array(hrir)


def get_geo(speaker_dist, spaker_to_head, ear_dist):
    s = speaker_dist / 2
    r = ear_dist / 2
    a = np.sqrt(spaker_to_head ** 2 - s ** 2)  # distance between head and the line of stereo speakers
    theta = np.degrees(np.arccos(a / speaker_to_head))  # angle between head and two speakers

    l1 = np.sqrt(a ** 2 + (s - r) ** 2)  # ipsilateral ear
    l2 = np.sqrt(a ** 2 + (s + r) ** 2)  # contralateral ear
    delta_l = l2 - l1

    delta_t = float(delta_l / SPEED)  # time delay between both ears

    return {'l1': l1, 'l2': l2, 'delta_l': delta_l, 'delta_t': delta_t, 'theta': theta}


if __name__ == '__main__':
    # prepare hrir
    data_ir, source_coordinates, receiver_coordinates = pf.io.read_sofa(
        'hrtf/IRC_1003_C_44100.sofa')
    hrir = get_ir(source_coordinates, data_ir)
    hrir_sr = 44100

    # prepare stereo audio
    file = './audio/guzheng.wav'
    sig, sr = sf.read(file)
    sig = librosa.resample(sig, orig_sr=sr, target_sr=hrir_sr, axis=0)
    sig = sig.T

    # personal configurations in meters
    speaker_dist = 0.5
    speaker_to_head = 0.5
    ear_dist = 0.15
    geometry = get_geo(speaker_dist, speaker_to_head, ear_dist)


    # core function
    target = CrosstalkCancel(sig, sr, geometry)
    cancelled_sig = target.xtc_process(hrir)

    # write out
    sf.write('xtc.wav', cancelled_sig.T, hrir_sr)
