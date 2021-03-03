import numpy as np
import matplotlib.pyplot as plt
from Lab1 import point_source
from scipy.signal import convolve2d

# Generate 2D Wave Field pattern extending it to y = -60 labmda data hyper-plane.
def visualize_wavefield(greens_amp=None, wavelength=1, relative_radius=30, sample_spacing= 1/4, pat_title=None, spec_title=None, data_plane=None):

    # Plot wave field pattern
    first_pt_flag = True
    for x,y in np.array([[0,10], [10,0], [0,-10], [-10,0], [-8,-6], [8,-6]]):
        wave_field_pattern = point_source.generate_wave_field_pattern(relative_radius, sample_spacing, wavelength,
                                                                      xshift=x, yshift=y, greens_amp=greens_amp, data_plane=data_plane)
        if first_pt_flag:
            wave_field_6pt_pattern = wave_field_pattern
            first_pt_flag = False
        else:
            wave_field_6pt_pattern += wave_field_pattern

    pattern_title = pat_title if pat_title else 'Source Wave-Field Pattern \nand Aperture Sampling'
    ax = point_source.plt_plot(np.abs(wave_field_6pt_pattern), title=pattern_title,
             xlabel='X position', ylabel='Y position')

    # Plot data plane at -60
    i, j = wave_field_6pt_pattern.shape
    y_v = (i-1)*np.ones(j)
    x_v = np.arange(j)
    plt.scatter(x_v, y_v, color='r', s=2)
    plt.show()

    return wave_field_6pt_pattern

def image_reconstruction(wave_field_at_receiver, radius, sample_spacing, wavelength, phase_only=False):
    # Plot wave field pattern

    wave_field_at_receiver = np.conjugate(wave_field_at_receiver)
    first_pt_flag = True
    for x, amplitude in enumerate(wave_field_at_receiver):
        wave_field_pattern = point_source.generate_wave_field_pattern(radius, sample_spacing, wavelength,
                                                                      xshift=x*sample_spacing -radius, yshift=-60,
                                                                      greens_amp=amplitude, data_plane=None, inverse=True,
                                                                      phase_only=phase_only)
        if first_pt_flag:
            reconstructed_image = wave_field_pattern
            first_pt_flag = False
        else:
            reconstructed_image += wave_field_pattern

    pattern_title = 'Reconstructed Source Region'
    reconstructed_image = np.conjugate(reconstructed_image)
    point_source.plt_plot(np.abs(reconstructed_image), title=pattern_title,
             xlabel='X position', ylabel='Y position')

    # Overlay pointsource locations
    for x, y in np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]]):
        x_idx = x/sample_spacing + (2*radius/sample_spacing)//2
        y_idx = -(y/sample_spacing - (2*radius/sample_spacing)//2)
        plt.scatter(x_idx,y_idx,color='r',s=3)

    # Plot wave field spectrum
    wave_field_spectrum = np.fft.fftshift(np.fft.fft2(reconstructed_image, s=(512, 512)))
    spectrum_title = 'Spectrum of Reconstructed Source Region'
    point_source.plt_plot(np.abs(wave_field_spectrum), title=spectrum_title,
             xlabel='$f_x$', ylabel='$f_y$')
    plt.show()

    # # Alternative: create wavefield by convolving aperture data with conjugate impulse response.
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

def normalize_wave_field(element):
    mag = np.linalg.norm(element)
    return mag

if __name__ == '__main__':
    wave_field_2d = visualize_wavefield(data_plane=-60)
    wave_field_at_receiver = wave_field_2d[-1, :]

    norm_wavefield = np.vectorize(normalize_wave_field)
    norm_wave_field_at_receiver = wave_field_at_receiver/norm_wavefield(wave_field_at_receiver)
    image_reconstruction(norm_wave_field_at_receiver, 30, 1/4, 1)

    image_reconstruction(norm_wave_field_at_receiver, 30, 1/4, 1, phase_only=True)
