import numpy as np
import matplotlib.pyplot as plt


class simulation(): # Single centerd transmitter, twin receivers
    def __init__(self, target_loc, distance_rx_t, num_freq, ref_wavelength=1):
        self.target_loc = target_loc
        self.distance_rx_tx = distance_rx_t
        self.num_freq = num_freq
        self.ref_wavelength = ref_wavelength

    def calc_distance(self, coord_1, coord_2):
        return np.sqrt( (coord_1[0] - coord_2[0])**2 + (coord_1[1] - coord_2[1])**2)

    def sim_data_collect(self, receiver_location):
        receiver_data = np.zeros(self.num_freq, dtype=np.complex)
        for i in range(self.num_freq):
            lam = self.num_freq * self.ref_wavelength / (i + self.num_freq)
            r_tx = self.calc_distance((0,0), self.target_loc)
            r_rx = self.calc_distance(receiver_location, self.target_loc)
            receiver_data[i] = np.exp(1j * 2 * np.pi * (r_tx + r_rx) / lam)
        return receiver_data

    def compute_range_profile(self, rx_data, rx_loc, target_window, scale=2, plot_range=False, plot_image=False):

        mult = 16 # Multiplier for higher resolution
        fft_bin = np.fft.fft(rx_data, self.num_freq * mult)
        range_v = np.linspace(0, self.num_freq, len(fft_bin))/scale
        self.range_v = abs(fft_bin/np.max(abs(fft_bin)))*range_v
        range_peak_idx = np.where(abs(fft_bin) == np.max(abs(fft_bin)))[0][0]
        range_peak = range_v.ravel()[range_peak_idx]


        if plot_range:
            plt.figure()
            plt.plot(range_v, abs(fft_bin))
            plt.annotate(fr"Peak at $r$ = {round(range_peak,2)}", xy=(range_peak, abs(fft_bin).ravel()[range_peak_idx]),
                         xytext=(range_peak+3, abs(fft_bin).ravel()[range_peak_idx]), arrowprops=dict(facecolor='red',
                                                           shrink=0.1), xycoords="data", )
            plt.xlabel(r"Range $r$")
            plt.ylabel("Likelihood of given range")
            print(f"Estimated Range from {rx_loc}: {range_peak}")
            print(f"Actual Range from {rx_loc}: {np.linalg.norm((self.target_loc[0]-rx_loc[0])+(self.target_loc[1]-rx_loc[1])*1j)}\n")

        if plot_image:
            x_ax = np.arange(target_window[0,0], target_window[0,1], 0.25 / 2)
            y_ax = np.arange(target_window[1,0], target_window[1,1], 0.25 / 2)
            X, Y = np.meshgrid(x_ax, y_ax)
            Y = np.rot90(Y, 2)
            image = np.zeros(X.shape).ravel()

            for i, xy in enumerate(zip(X.ravel(), Y.ravel())):
                r = np.linalg.norm([xy[0] - rx_loc[0], xy[1] - rx_loc[1]]) * mult
                pxl_val = 0.5 * (fft_bin[int(np.floor(r))] + fft_bin[int(np.ceil(r))])
                image[i] = abs(pxl_val)

            image = image.reshape(X.shape)

            return image / np.max(abs(image)), range_peak
        else:
            return range_peak

    def bearing_angle(self, rx_data):
        mult = 16
        fft_bin = np.fft.fftshift(np.fft.fft(rx_data, self.num_freq * mult))
        angle_v = np.linspace(-128/2, 128/2, len(fft_bin))/(2*self.distance_rx_tx)

        phase_peak_idx = np.where(abs(fft_bin) == np.max(abs(fft_bin)))[0][0]
        phase = angle_v.ravel()[phase_peak_idx]
        angle = 90 - np.degrees(np.arcsin(phase))

        plt.figure()
        idx_min = np.argmin(abs(angle_v+1))
        idx_max = np.argmin(abs(angle_v-1))
        plt.plot(90 - np.degrees(np.arcsin(angle_v[idx_min:idx_max+1])), abs(fft_bin[idx_min:idx_max+1]))
        plt.annotate(fr"Peak at $\theta$ = {round(angle,2)}", xy=(angle, abs(fft_bin).ravel()[phase_peak_idx]),
                       xytext=(100, 60), arrowprops=dict(facecolor='yellow',
                                                      shrink=0.05), xycoords="data", )
        plt.xlabel(r"Bearing Angle $\theta$")
        plt.ylabel("Magnitude")

        return angle

def main():
    rxtx_d = 1 # Distance between transmitter and receivers
    target_loc = (-6,14)
    win_rad = 30
    num_freq = 128

    target_window = np.array([[2*target_loc[0]-win_rad, 2*target_loc[0]+win_rad], [2*target_loc[1]-win_rad, 2*target_loc[1]+win_rad]])

    sim = simulation(target_loc, rxtx_d, num_freq)
    g1 = sim.sim_data_collect((-rxtx_d,0))
    g2 = sim.sim_data_collect((rxtx_d,0))

    im1, _ = sim.compute_range_profile(g1, (-rxtx_d, 0), target_window, plot_image=True, plot_range=True)
    im2, _ = sim.compute_range_profile(g2, (rxtx_d, 0), target_window, plot_image=True, plot_range=True)

    # Intersection of range profile from both receivers
    plt.figure()
    max_point = np.argmax(abs(im1 + im2))
    max_mat = np.zeros(im1.shape)
    max_mat.ravel()[max_point] = 0 # set to 10 to highlight predicted intersection.
    plt.imshow(abs(im1+im2+max_mat), extent=[target_window[0,0]/2, target_window[0,1]/2, target_window[1,0]/2, target_window[1,1]/2])
    plt.plot(target_loc[0], target_loc[1], 'r*')

    # Range profile of g1(n)g2(n) from transmitter POV
    target_window = np.array([[4 * target_loc[0] - win_rad, 4 * target_loc[0] + win_rad],
                              [4 * target_loc[1] - win_rad, 4 * target_loc[1] + win_rad]])
    im_comp, range_comp= sim.compute_range_profile(g1 * g2, (0, 0), target_window, scale=4, plot_image=True, plot_range=True)
    plt.figure()
    plt.imshow(abs(im_comp), extent=[target_window[0,0]/4, target_window[0,1]/4, target_window[1,0]/4, target_window[1,1]/4])
    plt.plot(target_loc[0], target_loc[1], 'r*')
    print(f"Estimated Range: {range_comp}")
    print(f"Actual Range: {np.linalg.norm(target_loc[0]+target_loc[1]*1j)}")

    # Bearing angle estimation from g1(n)g2*(n) from transmitter POV
    angle = sim.bearing_angle(g1 * np.conjugate(g2))
    print(f"Estimated Angle: {angle}")
    print(f"Actual Angle: {np.degrees(np.arctan2(target_loc[1],target_loc[0]))}")

    # Create polar plot
    plt.figure()
    theta = np.arange(0, 2 * np.pi, 0.01)
    plt.polar(theta, range_comp*np.ones(theta.shape), 'r', lw=3)
    plt.polar([np.deg2rad(angle),np.deg2rad(angle)], [0,range_comp+5], 'b', lw=3)
    plt.plot(np.deg2rad(angle), range_comp, 'yo')
    plt.annotate(fr"$\theta$ = {round(angle, 2)}, r = {round(range_comp, 2)}",
                xy=(np.deg2rad(angle), range_comp),  # theta, radius
                xytext=(0.65, 0.95),  # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom',
                )

    plt.show()

if __name__ == '__main__':
    main()