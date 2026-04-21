import numpy as np

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.matrix = np.zeros((height, width))