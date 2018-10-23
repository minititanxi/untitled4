
# import  csv

class Game:
    Fps_cpucore =[]
    Fps_gpu=[]
    Intensity_gpu=[]
    def fps(self,fps):#参数为帧数、两条sensitivity曲线、gpu_intensity:
        self.Fps=fps
    def fps_cpucore(self,Fps_cpucore):#参数为帧数、两条sensitivity曲线、gpu_intensity:
        self.Fps_cpucore = Fps_cpucore
    def fps_gpu(self,Fps_gpu):#参数为帧数、两条sensitivity曲线、gpu_intensity:
        self.Fps_gpu = Fps_gpu
    def intensity_gpu(self,intensity_gpu):#参数为帧数、两条sensitivity曲线、gpu_intensity:
        self.Intensity_gpu = intensity_gpu
    def game_number(self, game_num):  # 参数为帧数、两条sensitivity曲线、gpu_intensity:
        self.Game_number = game_num

class Data:

    def __init__(self,game,intensity_g,flag,cpu_core,colocated):#参数为game实例，固定维度的intensity,gpu1和2上的intensity vector:
        self.game=game
        self.intensity_g = intensity_g
        self.flag=flag
        self.cpu_core=cpu_core
        self.colocated=colocated



