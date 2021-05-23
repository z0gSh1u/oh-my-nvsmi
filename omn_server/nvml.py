# Wrapper for nvidia-ml-py, a Python port for NVML API.
# @see https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html

from pynvml import *

NVML_TEMPERATURE_GPU = 0
NVML_CUDA_DRIVER_VERSION_MAJOR = lambda version: str(version / 1000)


class NVMLWrapper:
    def __init__(self) -> None:
        nvmlInit()

        self.driver_version = nvmlSystemGetDriverVersion()
        self.cuda_version = NVML_CUDA_DRIVER_VERSION_MAJOR(nvmlSystemGetCudaDriverVersion())
        self.device_count = nvmlDeviceGetCount()
        self.devices_static = []
        self.devices_dynamic = []
        self._init_devices_static()
        self._refresh_devices_dynamic()

    def _init_devices_static(self):
        '''
          Initialize devices information that usually won't change along time.
        '''
        for index in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(index)
            device_name = nvmlDeviceGetName(handle)
            self.device_static.append({'index': index, 'device_name': device_name})

    def _refresh_devices_dynamic(self):
        '''
          Refresh devices information that often changes along time.
        '''
        for device in self.devices_static:
            dynamic = {}
            handle = nvmlDeviceGetHandleByIndex(device['index'])
            # GPU Memory
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            dynamic['memory_total'] = memory_info['total']
            dynamic['memory_free'] = memory_info['free']
            dynamic['memory_used'] = memory_info['used']
            # Temperature
            dynamic['temperature'] = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)  # in degrees C
            # GPU-Util
            dynamic['gpu_util'] = nvmlDeviceGetUtilizationRates(handle)  # percent
            self.devices_dynamic[device['index']] = dynamic

    def refresh_devices_dynamic(self):
        '''
          Manually refresh devices information that often changes along time.
        '''
        self._refresh_devices_dynamic()