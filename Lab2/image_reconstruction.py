import numpy as np
import matplotlib.pyplot as plt
from Lab1 import point_source
from Lab1.wave_field import Wave_Field
import time
from scipy.signal import convolve2d

#
# # Generate 2D Wave Field pattern extending it to y = -60 labmda data hyper-plane.
# def visualize_wavefield(greens_amp=None, wavelength=1, relative_radius=30, sample_spacing=1 / 4, pat_title=None,
#                         spec_title=None, data_plane=None):
#     # Plot wave field pattern
#     first_pt_flag = True
#     for x, y in np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]]):
#         wave_field_pattern = point_source.generate_wave_field_pattern(relative_radius, sample_spacing, wavelength,
#                                                                       xshift=x, yshift=y, greens_amp=greens_amp,
#                                                                       data_plane=data_plane)
#         if first_pt_flag:
#             wave_field_6pt_pattern = wave_field_pattern
#             first_pt_flag = False
#         else:
#             wave_field_6pt_pattern += wave_field_pattern
#
#     pattern_title = pat_title if pat_title else 'Source Wave-Field Pattern \nand Aperture Sampling'
#     ax = point_source.plt_plot(np.abs(wave_field_6pt_pattern), title=pattern_title,
#                                xlabel='X position', ylabel='Y position')
#
#     # Plot data plane at -60
#     i, j = wave_field_6pt_pattern.shape
#     y_v = (i - 1) * np.ones(j)
#     x_v = np.arange(j)
#     plt.scatter(x_v, y_v, color='r', s=2)
#     plt.show()
#
#     return wave_field_6pt_pattern
#
#
# def image_reconstruction2(wave_field_at_receiver, radius, sample_spacing, wavelength):
#     # Plot wave field pattern
#
#     wave_field_at_receiver = np.conjugate(wave_field_at_receiver)
#     first_pt_flag = True
#     for x, amplitude in enumerate(wave_field_at_receiver):
#         wave_field_pattern = point_source.generate_wave_field_pattern(radius, sample_spacing, wavelength,
#                                                                       xshift=x * sample_spacing - radius, yshift=-60,
#                                                                       greens_amp=amplitude, data_plane=None,
#                                                                       inverse=True)
#         if first_pt_flag:
#             reconstructed_image = wave_field_pattern
#             first_pt_flag = False
#         else:
#             reconstructed_image += wave_field_pattern
#
#     pattern_title = 'Reconstructed Source Region'
#     reconstructed_image = np.conjugate(reconstructed_image)
#     point_source.plt_plot(np.abs(reconstructed_image), title=pattern_title,
#                           xlabel='X position', ylabel='Y position')
#
#     # Overlay pointsource locations
#     for x, y in np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]]):
#         x_idx = x / sample_spacing + (2 * radius / sample_spacing) // 2
#         y_idx = -(y / sample_spacing - (2 * radius / sample_spacing) // 2)
#         plt.scatter(x_idx, y_idx, color='r', s=3)
#
#     # Plot wave field spectrum
#     wave_field_spectrum = np.fft.fftshift(np.fft.fft2(reconstructed_image, s=(512, 512)))
#     spectrum_title = 'Spectrum of Reconstructed Source Region'
#     point_source.plt_plot(np.abs(wave_field_spectrum), title=spectrum_title,
#                           xlabel='$f_x$', ylabel='$f_y$')
#     plt.show()
#
#     # # Alternative: create wavefield by convolving aperture data with conjugate impulse response.
#     # source = np.zeros((int(2*radius/sample_spacing) + 1,int(2*radius/sample_spacing) + 1))
#     # source[-1, :] = wave_field_at_receiver
#     # wave_field_pattern = point_source.generate_wave_field_pattern(3*radius, sample_spacing, wavelength=wavelength,
#     #                                                  greens_amp=None)
#     # wave_field = convolve2d(source, wave_field_pattern, mode='valid')
#     # wave_field = np.conj(wave_field)
#     # point_source.formatted_plot(np.abs(wave_field), title="Magnitude Distribution of \nReconstructed Source Region",
#     #          xlabel='X position', ylabel='Y position')
#     #
#     # plt.show()

class LinearReceiver():
    def __init__(self, y_int, xlim, slope=0, receiver_spacing=1):
        self.slope = slope
        self.y_int = y_int
        self.xlim = xlim
        self.receiver_spacing = receiver_spacing # TODO: incoporate difference

    def set_receiver_coordinates(self, source_WF):
        # Sample wave field at receiver (y = mx + b)
        receiver_length = int((self.xlim[1] - self.xlim[0]) / source_WF.sample_spacing + 1)
        # receiver_length = int((self.xlim[1] - self.xlim[0]) / self.receiver_spacing + 1)

        x_v = np.linspace(self.xlim[0], self.xlim[1], receiver_length)
        y_v = self.slope * x_v + self.y_int
        source_WF.receiver_coordinates = np.array([x_v, y_v]).transpose()
        self.receiver_coordinates = np.array([x_v, y_v]).transpose()

    def sample_wave_field(self, sources_xy_coordinates, source_WF, wavelength):
        # Sampled wavefield from simulated source distribution
        self.set_receiver_coordinates(source_WF)
        source_WF.generate_composite_wave_field_pattern(sources_xy_coordinates, wavelength=wavelength,
                                                        sample_mode=True,
                                                        sample_coordinates=source_WF.receiver_coordinates, plot=False)
    # def set_receiver_data(self, data):
    #     self.

    def reconstruct_image(self, source_WF, wavelength, pattern_title = "Reconstructed Source Region", simulation=True, xlim=None, ylim=None):

        # Reconstruct the source region with backpropagation
        if simulation:
            assert(hasattr(source_WF, 'receiver_coordinates'))
            assert(hasattr(source_WF, 'sampled_wavefield'))

            wave_field_at_receiver = np.conjugate(source_WF.sampled_wavefield)
            # wave_field_at_receiver = source_WF.sampled_wavefield
            self.reconstructed_wavefield = Wave_Field(source_WF.radius, source_WF.sample_spacing, phase_only=source_WF.phase_only)
            self.reconstructed_wavefield.generate_composite_wave_field_pattern(source_WF.receiver_coordinates, wavelength,
                                                                               wave_field_at_receiver, plot=False, xlim=xlim, ylim=ylim)
            self.reconstructed_wavefield.composite_wave_field_pattern = np.conjugate(self.reconstructed_wavefield.composite_wave_field_pattern)
            ax = self.reconstructed_wavefield.formatted_plot(self.reconstructed_wavefield.composite_wave_field_pattern, pattern_title)
            return ax


    def reconstruct_spectrum(self, fft_size=512, title="Wave-Field Spectrum"):
        ax = self.reconstructed_wavefield.generate_wave_field_spectrum(fft_size=fft_size, title=title)
        return ax


# def reconstruct_image(wavelength=1):
#     sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
#     pattern_title = "6 Point Source Wave-field Pattern"
#     spectrum_title = "6 Point Source Wave-field Spectrum"
#     radius = 30
#     sample_spacing = 1 / 4
#
#     # Create field pattern
#     source_WF = Wave_Field(radius=radius, sample_spacing=sample_spacing)
#     receiver_length = int(2*radius/sample_spacing +1)
#     y_v = (-60) * np.ones(receiver_length)
#     x_v = np.linspace(-radius, radius, receiver_length)
#     receiver_coordinates = np.array([x_v, y_v]).transpose()
#     print(receiver_coordinates)
#     print(receiver_coordinates.shape)
#     source_WF.generate_composite_wave_field_pattern(sources_xy_coordinates, wavelength=wavelength,
#                                                     sample_mode=True,
#                                                     sample_coordinates=receiver_coordinates, plot=False)
#
#     # Reconstruct the source region with backpropagation
#     wave_field_at_receiver = np.conjugate(source_WF.sampled_wavefield)
#     reconstructed_image = Wave_Field(radius, sample_spacing)
#     pattern_title = "Reconstructed Source Region"
#     ax = reconstructed_image.generate_composite_wave_field_pattern(receiver_coordinates, 1,
#                                                                    wave_field_at_receiver,
#                                                                    plot=False)
#     reconstructed_image_WF = np.conjugate(reconstructed_image.composite_wave_field_pattern)
#     ax = reconstructed_image.formatted_plot(reconstructed_image_WF, pattern_title)

def main():
    receiver = LinearReceiver(y_int=-60, xlim=(-30,30), slope=0)

    sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
    radius = 30
    sample_spacing = 1/4
    wavelength = 1

    # receiver.set_receiver_data()
    source_WF = Wave_Field(radius=radius, sample_spacing=sample_spacing) # Create empty wavefield to back propagate
    receiver.sample_wave_field(sources_xy_coordinates, source_WF, wavelength)
    ax = receiver.reconstruct_image(source_WF, wavelength)
    ax = receiver.reconstruct_spectrum()
    return ax

if __name__ == '__main__':

    start_time = time.time()
    ax = main()
    duration = time.time()-start_time
    print(duration)
    plt.show()


#     input("stop")
#     # VIsualize wave-field pattern and sampilng at -60 lambda_0
#     sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
#     pattern_title = "6 Point Source Wave-field Pattern"
#     spectrum_title = "6 Point Source Wave-field Spectrum"
#     radius = 30
#     sample_spacing = 1 / 4
#     # Plot wave field pattern
#     source_WF = Wave_Field(radius=radius, sample_spacing=sample_spacing)
#     wave_field_pattern = source_WF.generate_composite_wave_field_pattern(sources_xy_coordinates, 1, xlim=(-30, 30),
#                                                                          ylim=(-60, 30),
#                                                                          pattern_title=pattern_title)
#     # Plot data plane at -90 lambda_0
#     i, j = source_WF.composite_wave_field_pattern.shape
#     y_v = (i - 1) * np.ones(j)
#     x_v = np.arange(j)
#     wave_field_pattern.scatter(x_v, y_v, color='r', s=2)
#     plt.show()
#
#     # Plot wave field spectrum
#     wave_field_spectrum_plot = source_WF.generate_wave_field_spectrum(title=spectrum_title)
#     plt.show()
#
#     # Sample wave field at receiver (y = -60 lambda_0)
#     i, j = source_WF.composite_wave_field_pattern.shape
#     y_v = (-60) * np.ones(j)
#     x_v = np.linspace(-radius, radius, j)
#     receiver_coordinates = np.array([x_v, y_v]).transpose()
#     print(receiver_coordinates)
#     print(receiver_coordinates.shape)
#     source_WF.generate_composite_wave_field_pattern(sources_xy_coordinates, 1,
#                                                     sample_mode=True,
#                                                     sample_coordinates=receiver_coordinates, plot=False)
#     receiver_samples2 = source_WF.composite_wave_field_pattern[-1, :]
#     assert(np.allclose(receiver_samples2,source_WF.sampled_wavefield))
#
#     # Reconstruct the source region with backpropagation
#     wave_field_at_receiver = np.conjugate(source_WF.sampled_wavefield)
#     reconstructed_wavefield = Wave_Field(radius, sample_spacing)
#     pattern_title = "Reconstructed Source Region"
#     ax = reconstructed_wavefield.generate_composite_wave_field_pattern(receiver_coordinates, 1,
#                                                                    wave_field_at_receiver,
#                                                                    plot=False)
#     reconstructed_image = np.conjugate(reconstructed_wavefield.composite_wave_field_pattern)
#     ax = reconstructed_wavefield.formatted_plot(reconstructed_image, pattern_title)
#
#     # Overlay pointsource locations
#     for x, y in np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]]):
#         x_idx = x / sample_spacing + (2 * radius / sample_spacing) // 2
#         y_idx = -(y / sample_spacing - (2 * radius / sample_spacing) // 2)
#         plt.scatter(x_idx, y_idx, color='r', s=3)
#     plt.show()

