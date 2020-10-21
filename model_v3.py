import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from pprint import pprint
import pandas as pd


class GazeModel:
    def __init__(self, width: int = 1440, height: int = 1600):
        self.width = width
        self.height = height
