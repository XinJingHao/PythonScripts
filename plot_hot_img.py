import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os, shutil
import time

# Tensorboard, 0<value<1
timenow = str(datetime.now())[0:-10]
timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
writepath = 'runs/'+timenow
if os.path.exists(writepath): shutil.rmtree(writepath)
writer = SummaryWriter(log_dir=writepath)

grid_map = np.zeros((366,366))
for _ in range(10000):
	x = np.random.randint(0,366)
	y = np.random.randint(0,366)
	grid_map[x,y] += 1
	writer.add_image('Grid Map', grid_map/grid_map.max(), dataformats='HW')
	time.sleep(0.25)

writer.close()




import matplotlib.pyplot as plt

# matplotlib.pyplot
data = np.random.rand(10, 10)*255  # 生成一个10x10的随机数矩阵作为示例数据
plt.imshow(data, cmap='OrRd') # https://blog.csdn.net/qq_45091685/article/details/127948412
plt.colorbar()  # 添加颜色条
plt.show()  # 显示图像