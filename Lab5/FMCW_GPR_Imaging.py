import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import os
from Lab1.wave_field import Wave_Field
from Lab2.image_reconstruction import LinearReceiver
from scipy.io import loadmat


def reconstruction(wavelength, receiver_spacing, sample_spacing, GPR_data):

    radius = 100*receiver_spacing - receiver_spacing/2
    receiver = LinearReceiver(y_int=0, xlim=(-radius, radius), slope=0, receiver_spacing=receiver_spacing)

    # Reconstruct image with given wavelength
    plot_tittle = f"Reconstructed Source Region\n for $\lambda = {round(wavelength,3)}\lambda_0$"
    ax = receiver.reconstruct_image(wavelength, radius, sample_spacing, receiver_data=GPR_data, phase_only=True, pattern_title=plot_tittle, xlim=(-radius, 0), ylim=(-0.6,0))

    return receiver.reconstructed_wavefield.composite_wave_field_pattern

    # Save image png and data npy matrix
    data_dir = os.path.join(os.path.dirname(__file__), 'data2')
    plt.savefig("{}/{}_lambda.png".format(os.path.join(data_dir, 'reconstructed_images'), wavelength))
    np.save("{}/{}_lambda".format(os.path.join(data_dir, 'reconstructed_image_data'), wavelength),
            receiver.reconstructed_wavefield.composite_wave_field_pattern)

    # # Reconstruct spectrum
    ax = receiver.reconstruct_spectrum(title=f"Wave Field Spectrum\n for $\lambda = {round(wavelength,3)}\lambda_0$")
    plt.savefig("{}/{}_lambda.png".format(os.path.join(data_dir, 'reconstructed_spectrum'), wavelength))
    np.save("{}/{}_lambda".format(os.path.join(data_dir, 'reconstructed_spectrum_data'), wavelength),
            receiver.reconstructed_wavefield.wave_field_spectrum)


def multi_frequency_reconstruction(wavelengths):
    with multiprocessing.Pool() as pool:
        pool.map(reconstruction, wavelengths)


def superimpose_reconstructions(directory, vid_size, reconstruction_snapshots=None, video_name='reconstructed_image_video', FPS=2):
    # Add reconstructed images from each wavelength and create video.
    first_im_flag = True
    superimposed_reconstructions = np.zeros((10, 10)) # Final reconstruction image initialization

    width = vid_size[0]
    height = vid_size[1]
    FPS = FPS

    fourcc = VideoWriter_fourcc(*'MP42')
    reconstructed_video = VideoWriter(os.path.join(directory, f"{video_name}.avi"), fourcc, float(FPS), (width, height), 0)

    for i, file in enumerate(sorted(os.listdir(directory))):
        if file.endswith(".npy"):
            print(os.path.join(directory, file))
            reconstruction = np.load(os.path.join(directory, file))
            if first_im_flag:
                superimposed_reconstructions = reconstruction
                first_im_flag = False
            else:
                superimposed_reconstructions += reconstruction

            if reconstruction_snapshots:
                if i+1 in reconstruction_snapshots:
                    wavelength = round(float(file.split('_')[0]), 3)
                    # fig, ax = plt.figure()
                    plt.imshow(np.abs(superimposed_reconstructions))
                    title = f"Reconstruction for $\lambda = {wavelength}\lambda_0$"
                    plt.title(title)
                    plt.savefig("{}/super_{}_lambda.png".format(directory,wavelength))

            # superimposed_reconstructions = (abs(superimposed_reconstructions)/np.max(superimposed_reconstructions))*255

            reconstructed_video.write((255*(np.abs(superimposed_reconstructions)/np.max(np.abs(superimposed_reconstructions)))).astype(np.uint8))

    reconstructed_video.release()

    plt.imshow(np.abs(superimposed_reconstructions))
    plt.title(video_name)
    plt.show()


if __name__ == "__main__":

    input = int(input(
        "Options: \n1) Generate reconstruction data\n2) Generate reconstruction video\nnote: Must run 1) before 2)\n\nInput Option: "))

    if input == 1:

        # Spacing in m, Wavelength in Ghz
        frequencies = loadmat('gpr_data.mat')['f'].astype(np.float).squeeze()
        permitivity = 6
        velocity = 3*(10**8) / np.sqrt(permitivity)
        wavelengths = velocity / frequencies
        GPR_data = loadmat('gpr_data.mat')['F']

        first_w_flag = True
        for i in range(wavelengths.size):
            if first_w_flag:
                final_im = reconstruction(wavelengths[i], 0.0213, 0.0213/2, GPR_data[i, :])
                first_w_flag = False
            else:
                print(wavelengths[i])
                final_im += reconstruction(wavelengths[i], 0.0213, 0.0213/2, GPR_data[i, :])
            plt.close('all')
            print(f"Iteration : {i}")

            if (i+1)%32 == 0:
                plt.imshow(abs(final_im))
                plt.savefig(f'{i}.png')

        plt.imshow(abs(final_im))
        plt.show()

    elif input == 2:
        start_time = time.time()
        from cv2 import VideoWriter, VideoWriter_fourcc # Multiprocessing module raises error if imported before running
        root_dir = os.path.join(os.path.dirname(__file__), 'data/reconstructed_image_data')
        superimpose_reconstructions(root_dir, (241,241), reconstruction_snapshots=[10,20,30,40], video_name='reconstructed_image')
        duration = time.time() - start_time
        print(f"Duration {duration} seconds")