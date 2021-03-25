import numpy as np
import matplotlib.pyplot as plt
from Lab1 import point_source
from Lab1.wave_field import Wave_Field
import time
from Lab2.image_reconstruction import LinearReceiver


def normalize_wave_field(element):
    mag = np.linalg.norm(element)
    return mag


def phase_only_reconstruction(receiver, sources_xy_coordinates, radius, sample_spacing, wavelength, simulation=True, xlim=None, ylim=None):

    source_WF = Wave_Field(radius=radius, sample_spacing=sample_spacing, phase_only=True)
    receiver.sample_wave_field(sources_xy_coordinates, source_WF, wavelength)

    # Phase only reconstruction
    wave_field_at_receiver = receiver.sampled_wavefield
    norm_wavefield = np.vectorize(normalize_wave_field)
    norm_wave_field_at_receiver = wave_field_at_receiver / norm_wavefield(wave_field_at_receiver)
    receiver.sampled_wavefield = norm_wave_field_at_receiver


    ax = receiver.reconstruct_image(wavelength, radius, sample_spacing, phase_only=True, simulation=True, xlim=None, ylim=None)
    ax = receiver.reconstruct_spectrum()
    return ax

if __name__ == '__main__':

    start_time = time.time()
    receiver = LinearReceiver(y_int=-60, xlim=(-30, 30), slope=0, receiver_spacing=1)
    sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
    radius = 30
    sample_spacing = 1 / 4
    wavelength = 1
    ax = phase_only_reconstruction(receiver, sources_xy_coordinates, radius, sample_spacing, wavelength)
    duration = time.time()-start_time
    print(duration)
    plt.show()

'''

    # # Alternative: create wavefield by convolving aperture data with conjugate impulse response.
    # from scipy.signal import convolve2d
    # source = np.zeros((int(2*radius/sample_spacing) + 1,int(2*radius/sample_spacing) + 1))
    # source[-1, :] = wave_field_at_receiver
    # wave_field_pattern = point_source.generate_wave_field_pattern(3*radius, sample_spacing, wavelength=wavelength,
    #                                                  greens_amp=None)
    # wave_field = convolve2d(source, wave_field_pattern, mode='valid')
    # wave_field = np.conj(wave_field)
    # point_source.formatted_plot(np.abs(wave_field), title="Magnitude Distribution of \nReconstructed Source Region",
    #          xlabel='X position', ylabel='Y position')
    #
    # plt.show()


'''