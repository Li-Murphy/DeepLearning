import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
# 使用正则表达式 '\s+' 来指定分隔符为一个或多个空格

former_down = pd.read_csv('ooo_fit1/leaf.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
former_up = pd.read_csv('ooo_fit1/123.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)

# former_down = pd.read_csv('ooo/ooo_fit1_split1/blade1_ps.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
# former_up = pd.read_csv('ooo/ooo_fit1_split1/blade1_ss.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
# back_down = pd.read_csv('ooo/ooo_fit1_split1/blade2_translate8_ps.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
# back_up = pd.read_csv('ooo/ooo_fit1_split1/blade2_translate8_ss.dat',sep='\s+',names=['X', 'Y', 'Z'],skiprows=1)
x1 = former_down.loc[:,'X']
x2 = former_up.loc[:,'X']
# x3 = back_down.loc[:,'X']
# x4 = back_up.loc[:,'X']
#
y1 = former_down.loc[:,'Y']
y2 = former_up.loc[:,'Y']
# y3 = back_down.loc[:,'Y']
# y4 = back_up.loc[:,'Y']




plt.plot(x1,y1,color='black')
plt.plot(x2,y2,color='black')
#plt.plot(x3,y3,color='black')
#plt.plot(x4,y4,color='black')
plt.style.use('grayscale')

# 隐藏坐标轴
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
# 设置坐标轴范围
plt.xlim((0.47, 0.53))
plt.ylim((0.105, 0.143))
# 去除图框边缘
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# 移除图例和标题
plt.title('')
#plt.legend().set_visible(False)
plt.savefig('orginal.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
plt.show()
