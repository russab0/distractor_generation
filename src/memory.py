import subprocess as sp
import os
import nvidia_smi

import psutil


def get_gpu_usage(msg=''):
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print(f"{msg} | GPU | total {info.total}, free {info.free}, used {info.used} ")

    nvidia_smi.nvmlShutdown()


def get_cpu_usage(msg):
    # Calling psutil.cpu_precent() for 4 seconds
    print(f'{msg} | The CPU usage is: ', psutil.cpu_percent(4))


def get_ram_usage(msg):
    print(f'{msg} | RAM memory % used:', psutil.virtual_memory()[2])

