import numpy as np
import matplotlib.pyplot as plt
from Lab1.wave_field import Wave_Field
import time

class LinearReceiver():
    def __init__(self, y_int, xlim, slope=0, receiver_spacing=1):
        self.slope = slope
        self.y_int = y_int
        self.xlim = xlim
        self.receiver_spacing = receiver_spacing # TODO: incoporate difference

    def set_receiver_coordinates(self):
        # Sample wave field at receiver (y = mx + b)
        # receiver_length = int((self.xlim[1] - self.xlim[0]) / source_WF.sample_spacing + 1)
        receiver_length = int((self.xlim[1] - self.xlim[0]) / self.receiver_spacing + 1)

        x_v = np.linspace(self.xlim[0], self.xlim[1], receiver_length)
        y_v = self.slope * x_v + self.y_int
        # source_WF.receiver_coordinates = np.array([x_v, y_v]).transpose()
        self.receiver_coordinates = np.array([x_v, y_v]).transpose()

    def sample_wave_field(self, sources_xy_coordinates, source_WF, wavelength):
        # Sampled wavefield from simulated source distribution
        # self.set_receiver_coordinates(source_WF)
        self.set_receiver_coordinates()
        source_WF.generate_composite_wave_field_pattern(sources_xy_coordinates, wavelength, receiver=self,
                                                        sample_mode=True, plot=False) # sets sampled wf to receiver.sample_wavefield
    # def set_receiver_data(self, data):
    #     self.

    def reconstruct_image(self, wavelength, radius, sample_spacing, receiver_data=None, phase_only=True, pattern_title = "Reconstructed Source Region", xlim=None, ylim=None):

        # Reconstruct the source region with backpropagation
        if receiver_data is not None:
            self.set_receiver_coordinates()
            self.sampled_wavefield = receiver_data


        assert(hasattr(self, 'receiver_coordinates'))
        assert(hasattr(self, 'sampled_wavefield'))

        wave_field_at_receiver = np.conjugate(self.sampled_wavefield)
        self.reconstructed_wavefield = Wave_Field(radius, sample_spacing, phase_only=phase_only)
        self.reconstructed_wavefield.generate_composite_wave_field_pattern(self.receiver_coordinates, wavelength,
                                                                           amplitude=wave_field_at_receiver, plot=False, xlim=xlim, ylim=ylim)
        self.reconstructed_wavefield.composite_wave_field_pattern = np.conjugate(self.reconstructed_wavefield.composite_wave_field_pattern)
        ax = self.reconstructed_wavefield.formatted_plot(self.reconstructed_wavefield.composite_wave_field_pattern, pattern_title)
        return ax


    def reconstruct_spectrum(self, fft_size=512, title="Wave-Field Spectrum"):
        ax = self.reconstructed_wavefield.generate_wave_field_spectrum(fft_size=fft_size, title=title)
        return ax

def main():
    receiver = LinearReceiver(y_int=-60, xlim=(-30,30), slope=0, receiver_spacing=1)

    sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
    radius = 30
    sample_spacing = 1/4
    wavelength = 1

    # receiver.set_receiver_data()
    source_WF = Wave_Field(radius=radius, sample_spacing=sample_spacing) # Create empty wavefield to back propagate
    receiver.sample_wave_field(sources_xy_coordinates, source_WF, wavelength)
    ax = receiver.reconstruct_image(wavelength, radius, sample_spacing, phase_only=False)
    ax = receiver.reconstruct_spectrum()
    return ax

if __name__ == '__main__':

    start_time = time.time()
    ax = main()
    duration = time.time()-start_time
    print(duration)
    plt.show()


