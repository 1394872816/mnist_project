"""
设备监控模块 - 监控CPU/GPU使用情况
"""
import torch
import platform
import psutil


class DeviceMonitor:
    """设备监控器 - 实时监控GPU/CPU使用情况"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_gpu = torch.cuda.is_available()
        
        # 尝试导入GPU监控库
        self.pynvml_available = False
        if self.is_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.pynvml_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"Warning: pynvml init failed: {e}")
                print("  Using PyTorch built-in methods for GPU monitoring")
    
    def get_device_info(self):
        """获取设备详细信息"""
        info = {
            'device_type': 'GPU' if self.is_gpu else 'CPU',
            'device': self.device,
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'system': platform.system(),
        }
        
        if self.is_gpu:
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_capability': torch.cuda.get_device_capability(0),
            })
            
            # 获取GPU总内存
            if self.pynvml_available:
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                info['gpu_total_memory'] = mem_info.total / 1024**3  # GB
            else:
                info['gpu_total_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            info.update({
                'cpu_name': platform.processor(),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'ram_total': psutil.virtual_memory().total / 1024**3,  # GB
            })
            
        return info
    
    def get_gpu_usage(self):
        """获取当前GPU使用情况"""
        if not self.is_gpu:
            return None
            
        usage = {}
        
        if self.pynvml_available:
            # 使用pynvml获取详细信息
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            usage['memory_used'] = mem_info.used / 1024**3  # GB
            usage['memory_total'] = mem_info.total / 1024**3  # GB
            usage['memory_percent'] = (mem_info.used / mem_info.total) * 100
            usage['gpu_util'] = util.gpu  # GPU利用率 %
            
            # 获取温度
            try:
                temp = self.pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, self.pynvml.NVML_TEMPERATURE_GPU
                )
                usage['temperature'] = temp
            except:
                usage['temperature'] = None
                
            # 获取功耗
            try:
                power = self.pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # W
                usage['power'] = power
            except:
                usage['power'] = None
        else:
            # 使用PyTorch内置方法
            usage['memory_used'] = torch.cuda.memory_allocated() / 1024**3
            usage['memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
            usage['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage['memory_percent'] = (torch.cuda.memory_allocated() / 
                                       torch.cuda.get_device_properties(0).total_memory) * 100
            
        return usage
    
    def get_cpu_usage(self):
        """获取CPU使用情况"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_used': psutil.virtual_memory().used / 1024**3,
            'ram_total': psutil.virtual_memory().total / 1024**3,
            'ram_percent': psutil.virtual_memory().percent,
        }
    
    def format_usage_string(self):
        """格式化使用情况为字符串"""
        if self.is_gpu:
            gpu_usage = self.get_gpu_usage()
            if gpu_usage:
                s = f"GPU: {gpu_usage['memory_used']:.1f}/{gpu_usage['memory_total']:.1f}GB"
                if 'gpu_util' in gpu_usage:
                    s += f", Util: {gpu_usage['gpu_util']}%"
                if gpu_usage.get('temperature'):
                    s += f", Temp: {gpu_usage['temperature']}C"
                return s
        
        # CPU信息
        cpu_usage = self.get_cpu_usage()
        return f"CPU: {cpu_usage['cpu_percent']:.1f}%, RAM: {cpu_usage['ram_used']:.1f}/{cpu_usage['ram_total']:.1f}GB"
    
    def print_device_info(self):
        """打印设备信息"""
        info = self.get_device_info()
        
        print("\n" + "=" * 60)
        print("Device Information")
        print("=" * 60)
        
        if self.is_gpu:
            print(f"  Device Type  : GPU (CUDA)")
            print(f"  GPU Name     : {info['gpu_name']}")
            print(f"  GPU Count    : {info['gpu_count']}")
            print(f"  GPU Memory   : {info['gpu_total_memory']:.2f} GB")
            print(f"  CUDA Version : {info['cuda_version']}")
            print(f"  cuDNN Version: {info['cudnn_version']}")
            print(f"  Compute Cap  : {info['gpu_capability']}")
        else:
            print(f"  Device Type  : CPU")
            print(f"  CPU Name     : {info['cpu_name']}")
            print(f"  Cores/Threads: {info['cpu_cores']}/{info['cpu_threads']}")
            print(f"  RAM Size     : {info['ram_total']:.2f} GB")
            
        print(f"  Python       : {info['python_version']}")
        print(f"  PyTorch      : {info['pytorch_version']}")
        print(f"  OS           : {info['system']}")
        print("=" * 60 + "\n")
        
    def cleanup(self):
        """清理资源"""
        if self.pynvml_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass