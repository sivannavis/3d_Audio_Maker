"""
This script takes in mono audio file "Mister Magic" by Grover Washington, Jr.
and reproduce a 7.1.4 system surround sound effect in two methods.

Author: Sivan Ding (sivan.d@nyu.edu)

The 7.1.4 layout configuration is adopted from https://www.dolby.com/about/support/guide/speaker-setup-guides/7.1.4-overhead-speaker-setup-guide/

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
import soundfile as sf

mpl.use('TkAgg')


############ Virtual Surround Sound ##################
def vss(hrir, audio):
    surround_sound = {}
    surround_sound['left'] = []
    surround_sound['right'] = []
    for loc, ir in hrir.items():
        surround_sound['left'].append(np.convolve(audio[:, 0], ir[0, :]))
        surround_sound['right'].append(np.convolve(audio[:, 1], ir[1, :]))

    binaural_downmix = surround_2_bin(surround_sound, audio)

    return binaural_downmix


def object_vss(hrir, music):
    surround_sound = {}
    surround_sound['left'] = []
    surround_sound['right'] = []
    # guitar in the front (left, right, center)
    surround_sound['left'].append(np.convolve(music['guitar_l'], hrir['ls'][0, :]))
    surround_sound['left'].append(np.convolve(music['guitar_l'], hrir['rs'][0, :]))
    surround_sound['left'].append(np.convolve(music['guitar_l'], hrir['cs'][0, :]))
    surround_sound['right'].append(np.convolve(music['guitar_r'], hrir['ls'][1, :]))
    surround_sound['right'].append(np.convolve(music['guitar_r'], hrir['rs'][1, :]))
    surround_sound['right'].append(np.convolve(music['guitar_r'], hrir['cs'][1, :]))
    # drum in the back (left/right rear surround)
    surround_sound['left'].append(np.convolve(music['drum_l'], hrir['lrss'][0, :]))
    surround_sound['left'].append(np.convolve(music['drum_l'], hrir['rrss'][0, :]))
    surround_sound['right'].append(np.convolve(music['drum_r'], hrir['lrss'][1, :]))
    surround_sound['right'].append(np.convolve(music['drum_r'], hrir['rrss'][1, :]))
    # bass in subwoofer
    surround_sound['left'].append(np.convolve(music['bass_l'], hrir['sub'][0, :]))
    surround_sound['right'].append(np.convolve(music['bass_r'], hrir['sub'][1, :]))
    # other in center overhead and side (right/left surround, top rear overhead)
    surround_sound['left'].append(np.convolve(music['other_l'], hrir['lss'][0, :]))
    surround_sound['left'].append(np.convolve(music['other_l'], hrir['rss'][0, :]))
    surround_sound['left'].append(np.convolve(music['other_l'], hrir['ltfos'][0, :]))
    surround_sound['left'].append(np.convolve(music['other_l'], hrir['rtfo'][0, :]))
    surround_sound['left'].append(np.convolve(music['other_l'], hrir['ltros'][0, :]))
    surround_sound['left'].append(np.convolve(music['other_l'], hrir['rtros'][0, :]))
    surround_sound['right'].append(np.convolve(music['other_r'], hrir['lss'][1, :]))
    surround_sound['right'].append(np.convolve(music['other_r'], hrir['rss'][1, :]))
    surround_sound['right'].append(np.convolve(music['other_r'], hrir['ltfos'][1, :]))
    surround_sound['left'].append(np.convolve(music['other_r'], hrir['rtfo'][1, :]))
    surround_sound['left'].append(np.convolve(music['other_r'], hrir['ltros'][1, :]))
    surround_sound['left'].append(np.convolve(music['other_r'], hrir['rtros'][1, :]))

    binaural_downmix = surround_2_bin(surround_sound, audio)

    return binaural_downmix


def surround_2_bin(surround_sound, audio):
    # summing up channel and normalize value from 0~surround_max to 0~audio_max
    binaural_downmix = np.vstack((sum(surround_sound['left']), sum(surround_sound['right'])))
    binaural_downmix = binaural_downmix * np.amax(audio) / np.amax(binaural_downmix)

    return binaural_downmix


############ Stereo downmix ###############
def stereo_mix(audio):
    """
    The downmixing principle of this function refers to https://professionalsupport.dolby.com/s/article/How-do-the-5-1-and-Stereo-downmix-settings-work?language=en_US
    From 7.1 to 5.1:
    Ls = Lss + (–1.2 dB × Lrs) + (–6.2 dB × Rrs)
    Rs = Rss + (–6.2 dB × Lrs) + (–1.2 dB × Rrs)
    From 5.1 to stereo:
    Lt = L + (–3 dB × C) – (–1.2 dB × Ls) – (–6.2 dB × Rs)
    Rt = R + (–3 dB × C) + (–6.2 dB × Ls) + (–1.2 dB × Rs)
    And let's assume left channels are put to left speakers, right channels in right speakers and left at center too.

    :param audio:
    :return:
    """
    left = audio[:, 0]
    right = audio[:, 1]
    # downmix 7.1 to 5.1
    ls = left + minus_db(left, 1.2) + minus_db(left, 6.2)
    rs = right + minus_db(right, 6.2) + minus_db(right, 1.2)

    # downmix 5.1 to stereo
    l = left + minus_db(left, 3) - minus_db(ls, 1.2) - minus_db(rs, 6.2)
    r = right + minus_db(left, 3) - minus_db(ls, 6.2) - minus_db(rs, 1.2)

    stereo = np.vstack((l, r))
    stereo = stereo * np.amax(audio) / np.amax(stereo)
    return stereo


############ utils ###############
def db_to_amp(x):
    return 10 ** (x / 20)


def minus_db(signal, db):
    out = signal * ((np.amax(signal) - db_to_amp(db)) / np.amax(signal))
    return out


def test(source_coordinates, data_ir):
    # inspecting HRIR
    index, _ = source_coordinates.find_nearest_k(
        90, 0, 0.09, k=1, domain='sph', convention='top_elev', unit='deg', show=True)  # receiver radius is 0.09m
    _, mask = source_coordinates.find_slice(
        'elevation', unit='deg', value=0, show=True)
    pf.plot.time_freq(data_ir[index])
    with pf.plot.context():
        plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        angles = source_coordinates.get_sph('top_elev', 'deg')[mask, 0]
        ax, qm, cb = pf.plot.time_freq_2d(data_ir[mask, 0], indices=angles,
                                          cmap=mpl.cm.get_cmap(name='coolwarm'))
        ax[0].set_title("Left ear HRIR (Horizontal plane)")
        ax[0].set_xlabel("")
        ax[0].set_ylim(0, 3)
        qm[0].set_clim(-1.5, 1.5)
        ax[1].set_title("Left ear HRTFs (Horizontal plane)")
        ax[1].set_xlabel("Azimuth angle in degrees")
        ax[1].set_ylim(200, 20e3)
        qm[1].set_clim(-25, 25)
        plt.tight_layout()


def get_ir(source_coordinates, data_ir):
    # all measurements are done at distance 2.06m with (azimuth, elevation, radius)
    location = ['ls', 'rs', 'cs', 'sub', 'lss', 'rss', 'lrss', 'rrss', 'ltfos', 'rtfo', 'ltros', 'rtros']
    azimuth = [30, 330, 0, 20, 90, 270, 150, 210, 45, 315, 135, 225]
    elevation = [0, 0, 0, -30, 0, 0, 0, 0, 45, 45, 45, 45]
    # Left and right speakers ( 30, 0) (330, 0)
    # Center speaker (0, 0)
    # Subwoofer (20,  -30)
    # Left and right surround speakers (90, 0) (270, 0)
    # Left and right rear surround speakers (150, 0) (210, 0)
    # Left and right top front overhead speakers (45, 45) (315, 45)
    # Left and right top rear overhead speakers (135, 45) (225, 45)
    hrir = {}
    for i, loc in enumerate(location):
        index = source_coordinates.find_nearest_k(
            azimuth[i], elevation[i], 2.06, k=1, domain='sph', convention='top_elev', unit='deg')[0]
        print(loc, index, source_coordinates.cartesian[index])
        hrir[loc] = data_ir[index].time.T
    return hrir


def get_music():
    # audio separation is done by studio.gaudiolab.io
    time_len = 2 * 60  # truncate all instrument to 2 min

    music = {}
    for item in ['bass', 'drum', 'guitar', 'other']:
        instr, sr = sf.read(item + '.mp3')
        music[item + '_l'] = instr[:time_len * sr, 0]
        music[item + '_r'] = instr[:time_len * sr, 1]

    return music


if __name__ == '__main__':
    ###### prepare multi-channel audio
    audio, sr = sf.read('Mister Magic.flac')
    time_len = 2 * 60  # truncate all instrument to 2 min
    audio = audio[:time_len * sr, :]

    ###### prepare two channel HRTF at different locations (azimuth, elevation) of dolby 7.1.4 layout:
    data_ir, source_coordinates, receiver_coordinates = pf.io.read_sofa(
        'IRC_1003_C_44100.sofa')
    hrir = get_ir(source_coordinates, data_ir)

    ###### choose generating mode
    modes = ['test', 'vss', 'binaural', 'vss_music']
    mode = modes[2]

    if mode == 'vss':
        # reproducing vss
        reproduced_audio = vss(hrir, audio).T

    elif mode == 'binaural':
        # reproducing binaural mix
        reproduced_audio = stereo_mix(audio).T

    elif mode == 'vss_music':
        music = get_music()
        reproduced_audio = object_vss(hrir, music).T

    elif mode == 'test':
        test(source_coordinates, data_ir)

    else:
        raise NotImplementedError("Mode not defined yet.")

    sf.write('mixture_{}.wav'.format(mode), reproduced_audio, samplerate=sr)
