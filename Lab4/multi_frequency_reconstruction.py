import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import os
from Lab1.wave_field import Wave_Field
from Lab2.image_reconstruction import LinearReceiver


def reconstruction(wavelength):

    receiver = LinearReceiver(y_int=-60, xlim=(-30, 30), slope=0)

    sources_xy_coordinates = np.array([[0, 10], [10, 0], [0, -10], [-10, 0], [-8, -6], [8, -6]])
    radius = 30
    sample_spacing = 1 / 4
    source_WF = Wave_Field(radius=radius, sample_spacing=sample_spacing)
    receiver.sample_wave_field(sources_xy_coordinates, source_WF, wavelength)

    # Reconstruct image with given wavelength
    ax = receiver.reconstruct_image(source_WF, wavelength, pattern_title=f"Reconstructed Source Region\n for $\lambda = {round(wavelength,3)}\lambda_0$")
    # Save image png and data npy matrix
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
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

    wavelengths = 40 / (np.arange(1,41) + 20)
    start_time = time.time()

    # multi_frequency_reconstruction(wavelengths)

    from cv2 import VideoWriter, VideoWriter_fourcc # Multiprocessing module raises error if imported before running
    root_dir = os.path.join(os.path.dirname(__file__), 'data/reconstructed_image_data')
    superimpose_reconstructions(root_dir, (241,241), reconstruction_snapshots=[10,20,30,40], video_name='reconstructed_image')
    root_dir = os.path.join(os.path.dirname(__file__), 'data/reconstructed_spectrum_data')
    superimpose_reconstructions(root_dir, (512,512), reconstruction_snapshots=[10,20,30,40], video_name='reconstructed_spectrum')
    duration = time.time() - start_time
    print(f"Duration {duration} seconds")