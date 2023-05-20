import time
import numpy as np
import torch

'''测试不同采样函数的运行时间

--Result--

np_replaceFalse:
execution_time: 0.8837189674377441


np_replaceTrue:
execution_time: 0.7762010097503662


torch_replaceFalse_cpu:
execution_time: 0.5715374946594238


torch_replaceTrue_cpu:
execution_time: 0.15773844718933105


torch_replaceFalse_gpu:
execution_time: 0.08635497093200684


torch_replaceTrue_gpu:
execution_time: 0.005271196365356445

'''


buffer_size = int(1e6)
idx_list = np.arange(buffer_size)
p = np.random.random(buffer_size)
p_torch_cpu = torch.from_numpy(p)
p_torch_gpu = torch.from_numpy(p).to(torch.device('cuda:0'))

def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始时间
        for _ in range(100):
            result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录函数结束时间
        execution_time = end_time - start_time  # 计算函数执行时间
        print('execution_time:', execution_time)
        print('\n')
        return result
    return wrapper

# ---------------------------------------------------------
@calculate_time
def np_replaceFalse(p):
    idx = np.random.choice(a=idx_list, size=256, replace=False, p=(p/p.sum()))

print('np_replaceFalse:')
np_replaceFalse(p)


# ---------------------------------------------------------
@calculate_time
def np_replaceTrue(p):
    idx = np.random.choice(a=idx_list, size=256, replace=True, p=(p/p.sum()))

print('np_replaceTrue:')
np_replaceTrue(p)



# ---------------------------------------------------------
@calculate_time
def torch_replaceFalse_cpu(p_torch_cpu):
    idx = torch.multinomial(p_torch_cpu, num_samples=256, replacement=False)

print('torch_replaceFalse_cpu:')
torch_replaceFalse_cpu(p_torch_cpu)


# ---------------------------------------------------------
@calculate_time
def torch_replaceTrue_cpu(p_torch_cpu):
    idx = torch.multinomial(p_torch_cpu, num_samples=256, replacement=True)

print('torch_replaceTrue_cpu:')
torch_replaceTrue_cpu(p_torch_cpu)



# ---------------------------------------------------------
@calculate_time
def torch_replaceFalse_gpu(p_torch_gpu):
    idx = torch.multinomial(p_torch_gpu, num_samples=256, replacement=False)

print('torch_replaceFalse_gpu:')
torch_replaceFalse_gpu(p_torch_gpu)


# ---------------------------------------------------------
@calculate_time
def torch_replaceTrue_gpu(p_torch_gpu):
    idx = torch.multinomial(p_torch_gpu, num_samples=256, replacement=True)

print('torch_replaceTrue_gpu:')
torch_replaceTrue_gpu(p_torch_gpu)




