import numpy as np
from sympy import Poly
from sympy.abc import x
import sympy


def compute_h_s(a, b, s):
    """
    Computes complex transfer function on s
    :param a: denominator coefficients e.g. (b_0 * s**2 + b_1 * s + b_2)
    :param b: numerator coefficients e.g. (b_0 * s**2 + b_1 * s + b_2)
    :param s: np.array of input s
    :return: num/den transfer function
    """
    num = np.zeros_like(s, dtype=np.complex128)
    den = np.zeros_like(s, dtype=np.complex128)

    for n, b_i in enumerate(np.flip(b)):
        num += b_i * s ** n

    for n, a_i in enumerate(np.flip(a)):
        den += a_i * s ** n

    return num / den


def compute_h_z(a, b, z):
    """
    Computes complex transfer function on z
    :param a: denominator coefficients e.g. (b_0 + b_1 * z**(-1) + b_2 * z**(-2))
    :param b: numerator coefficients e.g. (b_0 + b_1 * z**(-1) + b_2 * z**(-2))
    :param z: np.array of input z
    :return: num/den transfer function
    """
    num = np.zeros_like(z, dtype=np.complex128)
    den = np.zeros_like(z, dtype=np.complex128)

    for n, b_i in enumerate(b):
        num += b_i * z ** (-n)

    for n, a_i in enumerate(a):
        den += a_i * z ** (-n)

    return num / den


def analog_to_digital(b, a, z, w0):
    """
    Compute h_z from analog s coefficients via bilinear transform
    c.f. https://ccrma.stanford.edu/~jos/pasp/Bilinear_Transformation.html
    :param a: denominator coefficients e.g. (b_0 * s**2 + b_1 * s + b_2)
    :param b: numerator coefficients e.g. (b_0 * s**2 + b_1 * s + b_2)
    :param z: z: np.array of input z
    :param w0: cycle frequency to map
    :return: num/den digital transfer function
    """
    s = (1/np.tan(w0/2)) * (1-z**(-1))/(1+z**(-1))
    return compute_h_s(a, b, s)


def compute_b_a(zeros, poles):
    """
    Compute B/A polynomial coefficients from zeros and poles
    :param zeros: complex zeros
    :param poles: complex poles
    :return: b, a coefficients
    """
    q = Poly(1, x)
    for zero in zeros:
        q = q.mul(Poly(x - zero, x)).mul(Poly(x - np.conjugate(zero), x))

    q = q.subs(sympy.I, 0)
    q = Poly(q, x)
    b = np.array(q.all_coeffs())
    b = np.round(b.astype(float), 4)

    p = Poly(1, x)
    for pole in poles:
        p = p.mul(Poly(x - pole, x)).mul(Poly(x - np.conjugate(pole), x))

    p = p.subs(sympy.I, 0)
    p = Poly(p, x)
    a = np.array(p.all_coeffs())
    a = np.round(a.astype(float), 4)

    return b, a


def filter_b_a(c, a, b):
    """
    Compute FIR and IIR outputs
    :param c: input signal
    :param a: numerator coefficients
    :param b: denominator coefficients
    :return: response signal
    """
    n = c.size
    warm_up = max(a.size, b.size)
    response = np.zeros(warm_up + n)
    c = np.pad(c, (warm_up, 1))
    for i in range(warm_up, n):
        response[i] += sum([c[i-j]*b[j] for j in range(len(b))])
        response[i] += sum([-response[i-j]*a[j] for j in range(1, len(a))])
    return response[warm_up:]