import numpy as np
import matplotlib.pyplot as plt
import json
import os


def dist(loc1, loc2):
    return np.sqrt((loc2[0] - loc1[0]) ** 2 + (loc2[1] - loc1[1]) ** 2)

def cap(s):
    return s[0].upper() + s[1:]