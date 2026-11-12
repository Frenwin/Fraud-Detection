"""Picklable transforms shared by research and serving."""
import numpy as np


def signed_log1p(a):
    return np.sign(a) * np.log1p(np.abs(a))
