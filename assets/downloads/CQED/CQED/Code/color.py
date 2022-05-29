# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:00:28 2021

@author: OTPS
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from PIL import Image 

img1 = Image.open(r"fit to merge.png", mode='r') 

img2 = Image.open(r"base to merge.png") 

img1.paste(img2, (0,0), mask = img2) 

img1.show()