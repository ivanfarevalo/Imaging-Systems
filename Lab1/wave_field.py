#!/usr/bin/env

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class Wave_Field():

    def __init__(self, radius, sample_spacing, phase_only=False):
        self.radius = radius
        self.sample_spacing = sample_spacing
        self.phase_only = phase_only

    class Point_Source():

        def __init__(self, radius, sample_spacing, wavelength=1, xshift=0, yshift=0, phase_only=False, amplitude=None):
            self.radius = radius
            self.sample_spacing = sample_spacing
            self.wavelength = wavelength
            self.xshift = xshift
            self.yshift = yshift
            self.phase_only = phase_only
            self.amplitude = amplitude
            self.v_wave_field_tf = np.vectorize(self.wave_field_tf, cache=False)

        def wave_field_tf(self, x, y):

            r = np.sqrt((x - self.xshift) ** 2 + (y - self.yshift) ** 2)

            # amplitude = 0

            if self.phase_only and self.amplitude:
                amplitude = self.amplitude
            elif self.phase_only:
                amplitude = 1
            elif self.amplitude:
                amplitude = self.amplitude * (1 / np.sqrt(1j * self.wavelength * r))
            else:
                amplitude = (1 / np.sqrt(1j * self.wavelength * r))

            return amplitude * np.exp(1j * 2 * np.pi * r / self.wavelength)

        def generate_wave_field_pattern(self, xlim=None, ylim=None):

            x = y = np.arange(-self.radius, self.radius + self.sample_spacing, self.sample_spacing)
            if xlim:
                assert (isinstance(xlim, (tuple, list, np.ndarray)))
                x = np.arange(xlim[0], xlim[1] + self.sample_spacing, self.sample_spacing)
            if ylim:
                assert (isinstance(ylim, (tuple, list, np.ndarray)))
                y = np.arange(ylim[0], ylim[1] + self.sample_spacing, self.sample_spacing)

            X, Y = np.meshgrid(x, y)
            wave_field_pattern = self.v_wave_field_tf(X, np.rot90(Y, 2))  # Y because of numpy indexing start from top left.

            return np.nan_to_num(wave_field_pattern)

        def sample_wave_field(self, coordinates):

            wave_field_pattern = self.v_wave_field_tf(coordinates[:,0], coordinates[:,1])
            return np.nan_to_num(wave_field_pattern)

    def generate_composite_wave_field_pattern(self, xy_coordinates, wavelength, amplitude=None, xlim=None, ylim=None,
                                              pattern_title="Wave Filed Pattern", sample_mode=False,
                                              sample_coordinates=None, plot=True):

        first_pt_flag = True
        for i, xy in enumerate(np.asarray(xy_coordinates).reshape(-1,2)):

            w = wavelength[i] if isinstance(wavelength, np.ndarray) else wavelength
            a = amplitude[i] if isinstance(amplitude, np.ndarray) else amplitude
            point_source = self.Point_Source(self.radius, self.sample_spacing, wavelength=w, xshift=xy[0], yshift=xy[1],
                                             amplitude=a, phase_only=self.phase_only)

            if sample_mode:
                assert(sample_coordinates is not None)
                sample_pattern = point_source.sample_wave_field(coordinates=sample_coordinates)
                if first_pt_flag:
                    self.sampled_wavefield = sample_pattern
                    first_pt_flag = False
                else:
                    self.sampled_wavefield += sample_pattern
            else:
                wave_field_pattern = point_source.generate_wave_field_pattern(xlim, ylim)

                if first_pt_flag:
                    self.composite_wave_field_pattern = wave_field_pattern
                    first_pt_flag = False
                else:
                    self.composite_wave_field_pattern += wave_field_pattern
        if plot:
            ax = self.formatted_plot(np.abs(self.composite_wave_field_pattern), title=pattern_title, xlabel='X position',
                                ylabel='Y position')
            return ax


    def generate_wave_field_spectrum(self, fft_size=512, title="Wave-Field Spectrum"):
        # Plot wave field spectrum
        self.wave_field_spectrum = np.fft.fftshift(np.fft.fft2(self.composite_wave_field_pattern, s=(fft_size, fft_size)))
        ax = self.formatted_plot(np.abs(self.wave_field_spectrum), title=title, xlabel='$f_x$', ylabel='$f_y$')
        return ax


    def formatted_plot(self, data, title='', xlabel='', ylabel=''):
        fig, ax1 = plt.subplots(1, 1, sharex=False)
        ax1.set_title(title)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,
            left=False,
            right=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
        ax1.imshow(np.abs(data))
        return ax1

