import numpy as np
from sympy import Poly
from sympy.abc import x
import sympy


def compute_h_z(a, b, z):
    """
    Computes complex transfer function on z
    :param a: denominator coefficients
    :param b: numerator coefficients
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