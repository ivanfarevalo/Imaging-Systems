import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from Lab1.wave_field import Wave_Field


def run_experiment(xy_coordinates, pat_title, spec_title, amplitude=None, wavelength=1, radius=30, sample_spacing=1 / 4):

    # Plot wave field pattern
    pt_source_WF = Wave_Field(radius, sample_spacing)
    wave_field_pattern = pt_source_WF.generate_composite_wave_field_pattern(xy_coordinates, wavelength, amplitude, pattern_title=pat_title)
    plt.show()

    # Plot wave field spectrum
    wave_field_spectrum = pt_source_WF.generate_wave_field_spectrum(title=spec_title)
    plt.show()



# def run_prob_2(greens_amp=None, wavelength=1,relative_radius=30,relative_sample_spacing=1/4, pat_title=None, spec_title=None, data_plane=None):

    # Plot wave field pattern
    # first_pt_flag = True
    # for x,y in np.array([[0,10], [10,0], [0,-10], [-10,0], [-8,-6], [8,-6]]):
    #     wave_field_pattern = generate_wave_field_pattern(relative_radius, relative_sample_spacing, wavelength,
    #                                                      xshift=x, yshift=y, greens_amp=greens_amp, data_plane=data_plane)
    #     if first_pt_flag:
    #         wave_field_6pt_pattern = wave_field_pattern
    #         first_pt_flag = False
    #     else:
    #         wave_field_6pt_pattern += wave_field_pattern
    #
    # pattern_title = pat_title if pat_title else '2D Coherent Wave-field Pattern \nfor 6 superimposed Point Sources'
    # formatted_plot(np.abs(wave_field_6pt_pattern), title=pattern_title,
    #          xlabel='X position', ylabel='Y position')
    #
    # # Plot wave field spectrum
    # wave_field_spectrum_plot = np.fft.fftshift(np.fft.fft2(wave_field_6pt_pattern, s=(512, 512)))
    # spectrum_title = spec_title if spec_title else '2D Fourier Spectrum of the Coherent Wave-field'
    # formatted_plot(np.abs(wave_field_spectrum_plot), title=spectrum_title,
    #          xlabel='$f_x$', ylabel='$f_y$')
    # plt.show()

    # # Alternative: create wavefield by convolving point sources with impulse response.
    # source = np.zeros(wave_field_pattern.shape)
    # for x, y in np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]]):
    # # for x, y in np.array([[0, 10], [0,-10],[10, 0]]):
    #     x_idx, y_idx = np.array(source.shape)//2 + np.array([x, -y])*relative_sample_spacing
    #     source[y_idx, x_idx]= 1
    #
    # wave_field_pattern = generate_wave_field_pattern(2*relative_radius, relative_sample_spacing, wavelength=wavelength,
    #                                                  greens_amp=greens_amp)
    # wave_field = convolve2d(source, wave_field_pattern, mode='same')
    # # wave_field = np.convolve(source.flatten(), wave_field_pattern.flatten(), mode='same')
    # # wave_field_spectrum_plot = np.fft.fftshift(np.fft.fft2(wave_field, s=(512, 512)))
    #
    # formatted_plot(np.abs(source), title="Source Distribution",
    #          xlabel='X position', ylabel='Y position')
    # plt.show()
    #
    # formatted_plot(np.abs(wave_field), title="Source Distribution",
    #          xlabel='X position', ylabel='Y position')
    # plt.show()



if __name__ == '__main__':
    input = int(input("Run Problem: "))
    if input == 1:
        # Single coherent point source at origin
        sources_xy_coordinates = np.array([[0, 0]])
        pattern_title = "Point Source Wave-field Pattern"
        spectrum_title = "Point Source Wave-field Spectrum"
        run_experiment(xy_coordinates = [0, 0], pat_title=pattern_title, spec_title=spectrum_title)
    elif input == 2:
        # Multiple coherent point sources at specified locations
        sources_xy_coordinates = np.array([[0,10], [10,0], [0,-10], [-10,0], [-8,-6], [8,-6]])
        pattern_title = "6 Point Source Wave-field Pattern"
        spectrum_title = "6 Point Source Wave-field Spectrum"
        run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title)
    elif input == 3:
        # Multiple coherent point sources at specified locations with different wavelengths
        sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
        sources_wavelengths = np.array([1,2,3,4,5,6])
        pattern_title = "6 Point Source Wave-field Pattern\n with different wavelengths"
        spectrum_title = "6 Point Source Wave-field Spectrum\n with different wavelengths"
        run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title, wavelength=sources_wavelengths)
    elif input == 4:
        # Single coherent point source at origin
        sources_xy_coordinates = np.array([[0, 0]])
        pattern_title = "Phase Only Point Source Wave-field Pattern"
        spectrum_title = "Phase Only Point Source Wave-field Spectrum"
        run_experiment(xy_coordinates=[0, 0], pat_title=pattern_title, spec_title=spectrum_title, amplitude=1)

        # Multiple coherent point sources at specified locations
        sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
        pattern_title = "Phase Only 6 Point Source Wave-field Pattern"
        spectrum_title = "Phase Only 6 Point Source Wave-field Spectrum"
        run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title, amplitude=1)

        # Multiple coherent point sources at specified locations with different wavelengths
        sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
        sources_wavelengths = np.array([1, 2, 3, 4, 5, 6])
        pattern_title = "Phase Only 6 Point Source Wave-field\n Pattern with different wavelengths"
        spectrum_title = "Phase Only 6 Point Source Wave-field\n Spectrum with different wavelengths"
        run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title,
                       wavelength=sources_wavelengths, amplitude=1)
    elif input == 5:

        for s in [1, 2, 4]:
            # Single coherent point source at origin
            sources_xy_coordinates = np.array([[0, 0]])
            pattern_title = "Point Source Wave-field Pattern with \n{}$\Lambda_0$ sample spacing".format(1/s)
            spectrum_title = "Point Source Wave-field Spectrum with \n{}$\Lambda_0$ sample spacing".format(1/s)
            run_experiment(xy_coordinates=[0, 0], pat_title=pattern_title, spec_title=spectrum_title, sample_spacing=1/s)

            # Multiple coherent point sources at specified locations
            sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
            pattern_title = "Point Source Wave-field Pattern with \n{}$\Lambda_0$ sample spacing".format(1/s)
            spectrum_title = "Point Source Wave-field Spectrum with \n{}$\Lambda_0$ sample spacing".format(1/s)
            run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title,
                           sample_spacing=1/s)

            # Multiple coherent point sources at specified locations with different wavelengths
            sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
            sources_wavelengths = np.array([1, 2, 3, 4, 5, 6])
            pattern_title = "Point Source Wave-field\n Pattern with different wavelengths and \n{}$\Lambda_0$ sample spacing".format(1/s)
            spectrum_title = "Point Source Wave-field\n Spectrum with different wavelengths and \n{}$\Lambda_0$ sample spacing".format(1/s)
            run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title,
                           wavelength=sources_wavelengths, sample_spacing=1/s)

    elif input == 6:

        for r in [15, 30, 60]:
            # Single coherent point source at origin
            sources_xy_coordinates = np.array([[0, 0]])
            pattern_title = "Point Source Wave-field Pattern \nwith Aperture Radius: {}$\lambda_0$".format(r)
            spectrum_title = "Point Source Wave-field Spectrum \nwith Aperture Radius: {}$\lambda_0$".format(r)
            run_experiment(xy_coordinates=[0, 0], pat_title=pattern_title, spec_title=spectrum_title,
                           radius=r)

            # Multiple coherent point sources at specified locations
            sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
            pattern_title = "6 Point Source Wave-field Pattern \nwith Aperture Radius: {}$\lambda_0$".format(r)
            spectrum_title = "6 Point Source Wave-field Spectrum \nwith Aperture Radius: {}$\lambda_0$".format(r)
            run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title,
                           radius=r)

            # Multiple coherent point sources at specified locations with different wavelengths
            sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
            sources_wavelengths = np.array([1, 2, 3, 4, 5, 6])
            pattern_title = "6 Point Source Wave-field\n Pattern with different wavelengths \nwith Aperture Radius: {}$\lambda_0$".format(r)
            spectrum_title = "6 Point Source Wave-field\n Spectrum with different wavelengths \nwith Aperture Radius: {}$\lambda_0$".format(r)
            run_experiment(xy_coordinates=sources_xy_coordinates, pat_title=pattern_title, spec_title=spectrum_title,
                           wavelength=sources_wavelengths, radius=r)
