import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import glob
from PIL import Image
# 使用正则表达式 '\s+' 来指定分隔符为一个或多个空格

former_down = pd.read_csv('Foil Data/split_B2_zz/former/blade1_ps.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
former_up = pd.read_csv('Foil Data/split_B2_zz/former/blade1_ss.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)

# 使用glob.glob()查找所有.dat文件
dat_files = glob.glob("Foil Data/split_B2_zz/back/*.dat")
num = len(dat_files)/2
back = []
former = []
i = 1
for file_path in dat_files:
    data = pd.read_csv(file_path,sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
    if i % 2 == 1:
        back.append(data)
    else:
        former.append((data))
    i = i + 1
up = np.vstack((former_up.loc[:,'X'],former_up.loc[:,'Y']))
down = np.vstack((former_down.loc[:,'X'],former_down.loc[:,'Y']))

for i in range(int(num)):
    Up = np.vstack((former[i].loc[:,'X'],former[i].loc[:,'Y']))
    Down = np.vstack((back[i].loc[:,'X'],back[i].loc[:,'Y']))
    plt.plot(up[0],up[1],color='black')
    plt.plot(down[0],down[1],color='black')
    plt.plot(Up[0],Up[1],color='black')
    plt.plot(Down[0],Down[1],color='black')

    plt.style.use('grayscale')
    # 设置坐标轴范围
    plt.xlim((0.47, 0.53))
    plt.ylim((0.105, 0.143))
    #隐藏坐标轴
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # 去除图框边缘
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # 移除图例和标题
    #plt.title('')

    plt.savefig(f'Foil{i+1}.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    plt.show()

