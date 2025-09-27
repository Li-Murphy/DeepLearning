import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sympy
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

img1 = Image.open('Data Picture/plot.png')
img1 = np.array(img1)
s1 = np.loadtxt('SHAP/plot1_Base.csv', delimiter=",")
H , W = s1.shape
new = np.zeros([H,W])
for i in range(H):
    for j in range(W):
        if s1[i,j] > 0.00005:
            if img1[i,j] == 255:
                new[i,j] = 0
            else:
                new[i,j] = 255
        else:
            new[i,j] = img1[i,j]
df = pd.DataFrame(new)
df.to_csv(f'foil.csv', index=False, header=False)
plt.imshow(new, cmap='Greys', interpolation='nearest')
#plt.imshow(img1, cmap='Greys', interpolation='nearest')
plt.show()