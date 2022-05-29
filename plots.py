from filters import compute_b_a, compute_h_z, filter_b_a
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = '/Users/alelouis/Library/Fonts/JetBrains Mono Regular Nerd Font Complete.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

def show_surface(extent, zeros, poles):
    lbo, ubo = extent
    x, y = np.meshgrid(np.linspace(lbo, ubo, 1000), np.linspace(lbo, ubo, 1000))
    b, a = compute_b_a(zeros, poles)
    h_z = compute_h_z(a, b, x + 1j * y)
    t = np.linspace(0, 2 * np.pi, 100)
    plt.figure(dpi=100, figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log(np.abs(h_z)), cmap='Greys', extent=[lbo, ubo, lbo, ubo])
    plt.plot(np.cos(t), np.sin(t), c='w', alpha=0.8, linewidth=0.5)
    plt.scatter(np.real(zeros), np.imag(zeros), c='r', s=0.5)
    plt.scatter(np.real(zeros), -np.imag(zeros), c='r', s=0.5)
    plt.scatter(np.real(poles), np.imag(poles), c='lime', s=0.5)
    plt.scatter(np.real(poles), -np.imag(poles), c='lime', s=0.5)
    plt.title('Modulus')
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(h_z), cmap='hsv', extent=[lbo, ubo, lbo, ubo])
    plt.plot(np.cos(t), np.sin(t), c='w', alpha=0.8, linewidth=0.5)
    plt.title('Phase')
    plt.show()


def show_path(z_path, zeros, poles):
    b, a = compute_b_a(zeros, poles)
    h_z_path = compute_h_z(a, b, z_path)
    plt.figure(dpi=100, figsize=(6, 3))
    plt.plot(np.linspace(-np.pi, np.pi, z_path.size), np.log(np.fft.fftshift(np.abs(h_z_path))), c='k')
    plt.xlim([-np.pi, np.pi])
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], labels=['-π', '-π/2', '0', 'π/2', 'π'])
    plt.grid(which='major', linewidth=0.5, linestyle=':')
    plt.title('Frequency response (unit circle path)')
    plt.show()


def show_ir(zeros, poles):
    b, a = compute_b_a(zeros, poles)
    impulse = np.zeros(30)
    impulse[0] = 1
    ir = filter_b_a(impulse, a, b)
    plt.figure(dpi=100, figsize=(6, 3))
    m, _, _ = plt.stem(ir, linefmt='k')
    m.set_markerfacecolor('none')
    m.set_markeredgecolor('k')
    plt.grid(which='major', linewidth=0.5, linestyle=':')
    plt.xlim([-1, ir.size])
    plt.title('Impulse response')
    plt.show()
