from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

PREDICT_SIZE=20


if len(argv)!=4:
    print("Provide exactly three argumnet: path to log, fps and coefficient for prediction")
    exit(1)

lines=[]
with open(argv[1]) as log:
    lines=[[*map(float,l.split())] for l in log]

fps=int(argv[2])
fph=fps*60*60

k=float(argv[3])

# Sample data
x,y_in,y_in_delayed,y_staff,y_out,mean_times=map(list,zip(*lines))

ndays=len(x)//(fph*24)

if ndays<=PREDICT_SIZE:
    print("Not enough data")
    exit(2)

current_day=[]
for hour in range(24):
    y_in_selected=[sum(y_in[fph*(24*d+hour):fph*(24*d+hour+1)])/fph for d in range(ndays)]

    data=[(y_in_selected[i:i+PREDICT_SIZE],y_in_selected[i+PREDICT_SIZE]) for i in range(ndays-PREDICT_SIZE)]

    model = LinearRegression()
    model.fit(*zip(*data))

    y_pred = model.predict([y_in_selected[-PREDICT_SIZE:]])
    current_day+=[*y_pred]

k/=sum(current_day)
current_day=[x*k for x in current_day]

print(*current_day)

plt.plot([*range(24)], current_day, label='Load', color='red')
plt.title('Current day predicted load')
plt.xlabel('Hour')
plt.ylabel('People')
plt.legend()
plt.grid()
plt.show()
