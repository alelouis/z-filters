import numpy as np
import matplotlib.pyplot as plt
from filters import analog_to_digital


class Equalizer:
    def __init__(self, Q, bands):
        """
        Initialize the peaking EQ
        :param Q: Quality factor
        :param bands: number of bands for EQ
        """
        self.Q = Q
        self.bands = bands

        step = np.pi / (bands + 1)
        self.gains_db = np.ones(bands)
        self.frequencies = np.arange(step, np.pi, step)
        self.eq_b = np.zeros((bands, 3))
        self.eq_a = np.zeros((bands, 3))
        self.update_coefficients()

    def update_coefficients(self):
        """
        Compute b and a analog coefficients.
        """
        for band in range(self.bands):
            A = 10 ** (self.gains_db[band] / 10)
            self.eq_b[band] = np.array([1, A / self.Q, 1])
            self.eq_a[band] = np.array([1, 1 / self.Q, 1])

    def plot_freq_response(self):
        """
        Show digital frequency response of all bands.
        """
        unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 1000))
        plt.figure(dpi=100)
        for band in range(self.bands):
            b, a = self.eq_b[band], self.eq_a[band]
            dft = analog_to_digital(b, a, unit_circle, self.frequencies[band])
            plt.plot(10 * np.log10(np.fft.fftshift(np.abs(dft))[dft.size // 2:]), c='k')
        plt.show()

    def set_gains(self, gains_db):
        """
        Replace bands gains
        :param gains_db: array of gains in dB for each band
        """
        self.gains_db = gains_db
        self.update_coefficients()
