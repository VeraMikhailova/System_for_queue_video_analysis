from operator import itemgetter
from sys import argv
import matplotlib.pyplot as plt

if len(argv)!=2:
    print("Provide exactly one argumnet: path to log")
    exit(1)

lines=[]
with open(argv[1]) as log:
    lines=[[*map(float,l.split())] for l in log]

# Sample data
x,y_in,y_in_delayed,y_staff,y_out=map(list,zip(*lines))

# Create a line plot for each dataset
#plt.plot(x, y_in, marker='o', label='In queue', color='red')
#plt.plot(x, y_staff, marker='s', label='Staff', color='blue')
#plt.plot(x, y_out, marker='^', label='Others', color='green')
plt.plot(x, y_in, label='In queue', color='red')
plt.plot(x, y_staff, label='Staff', color='blue')
plt.plot(x, y_out, label='Others', color='green')

# Add title and labels
plt.title('Log plot')
plt.xlabel('Time')
plt.ylabel('People')

# Show legend
plt.legend()

# Show grid
plt.grid()

# Display the plot
plt.show()

