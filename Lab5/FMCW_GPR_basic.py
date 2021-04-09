import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

def wave_field_tf(wavelength, amplitude, sample_x, sample_y, x, y):
    r = np.sqrt((x - sample_x) ** 2 + (y - sample_y) ** 2)

    return amplitude * np.exp(1j * 2 * np.pi * r / wavelength)

def generate_wave_field_pattern(wavelength, amplitude, sample_coordinate, spacing, xlim, ylim):

    x = np.arange(xlim[0], xlim[1] + spacing, spacing)
    y = np.arange(ylim[0], ylim[1] + spacing, spacing)

    X, Y = np.meshgrid(x, y)
    v_wave_field_tf = np.vectorize(wave_field_tf, cache=False)
    wave_field_pattern = v_wave_field_tf(wavelength, amplitude, sample_coordinate[0], sample_coordinate[1], X, np.rot90(Y, 2))  # Y because of numpy indexing start from top left.

    return np.nan_to_num(wave_field_pattern)


def main(wavelength, sample_spacing, GPR_data):
    rad = 100*sample_spacing - sample_spacing/2
    receiver_coordinates = np.linspace(-rad, rad, 200)

    first_flag = True
    for i, sample_coord in enumerate(receiver_coordinates):
        if first_flag:
            reconstruction = generate_wave_field_pattern(wavelength, GPR_data[i], (sample_coord, 0), sample_spacing/2, (-rad, rad), (-0.4, 0))
            first_flag = False
        else:
            reconstruction += generate_wave_field_pattern(wavelength, GPR_data[i], (sample_coord, 0), sample_spacing/2, (-rad, rad), (-0.4, 0))

    return reconstruction



if __name__ == "__main__":

    # Spacing in m, Wavelength in Ghz

    frequencies = loadmat('gpr_data.mat')['f'].astype(np.float).squeeze()
    permitivity = 6
    velocity = 3*(10**8) / np.sqrt(permitivity)
    wavelengths = velocity / frequencies

    start_time = time.time()
    GPR_data = loadmat('gpr_data.mat')['F']

    first_w_flag = True
    for i in range(wavelengths.size):
        if first_w_flag:
            final_im = main(wavelengths[i], 0.0213, GPR_data[i, :])
            first_w_flag = False
        else:
            final_im += main(wavelengths[i], 0.0213, GPR_data[i, :])

        print(f"Iteration : {i}")
        if (i+1)%32 == 0:

            plt.imshow(abs(final_im))
            plt.savefig(f'{i}.png')

    plt.imshow(abs(final_im))
    plt.show()
