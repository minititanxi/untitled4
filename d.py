import pandas as pd
import csv
from DataClass import Game
from DataClass import Data
import numpy as np
import copy
threshold=36
def print_to_train(data):
    temp=[]
    temp.append(float(data.game.Game_number))
    #temp.append(float(data.game.Fps[0]))
    temp.append(float(data.game.Fps_cpucore[0]))
    temp.append(float(data.game.Fps_cpucore[1]))
    temp.append(float(data.game.Fps_cpucore[2]))
    temp.append(float(data.game.Fps_cpucore[3]))
    temp.append(float(data.game.Fps_cpucore[4]))
    temp.append(float(data.game.Fps_cpucore[5]))
    temp.append(float(data.game.Fps_cpucore[6]))
    temp.append(float(data.game.Fps_cpucore[7]))
    temp.append(float(data.game.Fps_cpucore[8]))
    temp.append(float(data.game.Fps_cpucore[9]))
    temp.append(float(data.game.Fps_gpu[0]))
    temp.append(float(data.game.Fps_gpu[1]))
    temp.append(float(data.game.Fps_gpu[2]))
    temp.append(float(data.game.Fps_gpu[3]))
    temp.append(float(data.game.Fps_gpu[4]))
    temp.append(float(data.game.Fps_gpu[5]))
    temp.append(float(data.game.Fps_gpu[6]))
    temp.append(float(data.game.Fps_gpu[7]))
    temp.append(float(data.game.Fps_gpu[8]))
    temp.append(float(data.game.Fps_gpu[9]))
    temp.append(float(data.intensity_g[0]))
    temp.append(float(data.intensity_g[1]))
    temp.append(float(data.intensity_g[2]))
    temp.append(float(data.cpu_core))
    temp.append(float(data.flag))
    #writer.writerow(data.game.Fps,data.game.Fps_cpucore[0:9],data.game.Fps_gpu[0:9],data.intensity_g[0:2],data.cpu_core
               #     ,data.gpu1[0:2],data.gpu2[0:2],data.flag)

    return temp

temp_row_gpu = []
temp_row_cpu = []
temp_row_fps=[]
temp_row_intensity=[]
csv_reader = csv.reader(open('solo.csv', encoding='utf-8'))
f = open('2.csv','w', newline='')
writer = csv.writer(f)
for row in csv_reader:
    if row[0] == 'GPU_Load':
        temp_row_gpu.append(row[1:11])
        temp_row_intensity.append(row[12:13])
        writer.writerow(row[1:11])
    if row[0] == 'CPU_Core':
        temp_row_cpu.append(row[1:11])
        writer.writerow(row[1:11])
    if row[0] == 'fps':
        temp_row_fps.append(row[1:2])
        writer.writerow(row[1:2])

f.close()
print(temp_row_intensity)
vector_game=[]
for i in range(30):
    game = Game()
    game.fps(temp_row_fps[i])
    game.fps_gpu(temp_row_gpu[i])
    game.fps_cpucore(temp_row_cpu[i])
    game.intensity_gpu(temp_row_intensity[i])
    num=i
    num=num+1
    game.game_number(num)
    vector_game.append(game)

print(len(vector_game))

csv_reader1 = csv.reader(open('3.csv', encoding='utf-8'))
vector_data=[]
f = open('train.csv', 'w', newline='')
writer = csv.writer(f)
for row in csv_reader1:
    if row[4]!='':
        if row[14]!='':#CPU 3 0 gpu 3 0
            vector_intensity_ig1=[]
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[14])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(3)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19])>=threshold and int(row[21])>=threshold and int(row[23])>=threshold:
                flag=1
            else:
                flag=0
            if row[0]==row[12]:
                cpu_core=row[1]
            elif row[2]==row[12]:
                cpu_core=row[3]
            else:
                cpu_core=row[5]
            data = Data(vector_game[int(row[12])-1],vector_intensity_ig2,flag,cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)


            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13]) - 1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[14])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(3)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= 50 and int(row[21]) >= 50 and int(row[23]) >= 50:
                flag = 1
            else:
                flag = 0
            if row[0] == row[13]:
                cpu_core = row[1]
            elif row[2] == row[13]:
                cpu_core = row[3]
            else:
                cpu_core = row[5]
            data = Data(vector_game[int(row[13]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)


            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[14]) - 1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(3)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[14]:
                cpu_core = row[1]
            elif row[2] == row[14]:
                cpu_core = row[3]
            else:
                cpu_core = row[5]
            data = Data(vector_game[int(row[14]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
        else:#CPU 3 0 GPU 2 1

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(2)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= 50 and int(row[21]) >= 50 and int(row[23]) >= 50:
                flag = 1
            else:
                flag = 0
            if row[0] == row[12]:
                cpu_core = row[1]
            elif row[2] == row[12]:
                cpu_core = row[3]
            else:
                cpu_core = row[5]
            data = Data(vector_game[int(row[12]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(2)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[13]:
                cpu_core = row[1]
            elif row[2] == row[13]:
                cpu_core = row[3]
            else:
                cpu_core = row[5]
            data = Data(vector_game[int(row[13]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[15])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(1)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[15]:
                cpu_core = row[1]
            elif row[2] == row[15]:
                cpu_core = row[3]
            else:
                cpu_core = row[5]
            data = Data(vector_game[int(row[15]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

    else:
        if row[14] == '':#cpu2 1 GPU 2 1
            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(2)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[12]:
                cpu_core = row[1]
            elif row[2] == row[12]:
                cpu_core = row[3]
            else:
                cpu_core = row[7]
            data = Data(vector_game[int(row[12]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(2)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[13]:
                cpu_core = row[1]
            elif row[2] == row[13]:
                cpu_core = row[3]
            else:
                cpu_core = row[7]
            data = Data(vector_game[int(row[13]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[15])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(1)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[14]:
                cpu_core = row[1]
            elif row[2] == row[14]:
                cpu_core = row[3]
            else:
                cpu_core = row[7]
            data = Data(vector_game[int(row[15]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)
        else:#CPU 2 1 gpu 3 0
            vector_intensity_ig1=[]
            vector_intensity_ig1.append(float(vector_game[int(row[12]) - 1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[14])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(3)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19])>=threshold and int(row[21])>=threshold and int(row[23])>=threshold:
                flag=1
            else:
                flag=0
            if row[0]==row[12]:
                cpu_core=row[1]
            elif row[2]==row[12]:
                cpu_core=row[3]
            else:
                cpu_core=row[7]
            data = Data(vector_game[int(row[12]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13]) - 1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[14])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(3)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[13]:
                cpu_core = row[1]
            elif row[2] == row[13]:
                cpu_core = row[3]
            else:
                cpu_core = row[7]
            data = Data(vector_game[int(row[13]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)

            vector_intensity_ig1 = []
            vector_intensity_ig1.append(float(vector_game[int(row[12])-1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[13]) - 1].Intensity_gpu[0]))
            vector_intensity_ig1.append(float(vector_game[int(row[14])-1].Intensity_gpu[0]))
            vector_intensity_ig2 = []
            vector_intensity_ig2.append(3)
            vector_intensity_ig2.append(np.mean(vector_intensity_ig1))
            vector_intensity_ig2.append(np.var(vector_intensity_ig1))
            if int(row[19]) >= threshold and int(row[21]) >= threshold and int(row[23]) >= threshold:
                flag = 1
            else:
                flag = 0
            if row[0] == row[15]:
                cpu_core = row[1]
            elif row[2] == row[15]:
                cpu_core = row[3]
            else:
                cpu_core = row[7]
            data = Data(vector_game[int(row[14]) - 1], vector_intensity_ig2, flag, cpu_core)
            writer.writerow(print_to_train(data))
            vector_data.append(data)
            print(1)
f.close()

print(data)

















