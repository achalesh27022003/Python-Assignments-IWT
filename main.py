# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tkinter
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
# Q1
time = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
# returns evenly spaced values within a given interval. 0 & 1 are the default
#values of start & step args and they are optional.
# print(time)
np.sin(time)
np.cos(time)
# Q2
plt.rc('grid', linestyle="--", color='blue') # (this to customise grid on graph curve)
plt.suptitle("Sine and Cosine Curves : ", fontsize=14,
fontweight='bold') # ha = 'left', va = 'bottom', fontsize = 9 (ha :horizontal alignment.
plt.subplot(3, 1, 1, frameon=True,
facecolor='pink') # comma vali cheez dikhani hai yha pe (3, 1, 1) aur frame vali (frameon = True) & projection ='polar'
plt.title("Sine Curve", fontsize=9, fontweight='bold', color='red') # set a title for the axes.
plt.xlabel("Time", fontweight='bold', fontsize=9)
plt.ylabel("sin(t)", fontweight='bold', fontsize=9)
plt.grid(True)
plt.plot(time, np.sin(time), 'r') # yha dashed line vali cheezein btaani hai.
plt.subplot(3, 1, 3, frameon=True, facecolor='yellow')
plt.title("Cosine Curve", fontsize=8, fontweight='bold', color='green')
plt.xlabel("Time", fontweight='bold', fontsize=8)
plt.ylabel("cos(t)", fontweight='bold', fontsize=8)
plt.grid(False)
plt.plot(time, np.cos(time), 'g')
# plt.plot(time, np.sin(time), time, np.cos(time))
# Use above command to print in one graph.
plt.savefig("Figure1.png")
# Q3
amplitude1 = np.sin(time)
amplitude2 = np.cos(time)
# Q4
amplitude = np.array([time, amplitude1, amplitude2])
# print(amplitude)
# Q5
np.savetxt("assignment.csv", amplitude, delimiter=',', header='The Start',
footer='The End')
# Above command is use to save an array as text file. header and footer arw optional.
# Q6 & Q7
def openFile():
    data = np.loadtxt("assignment.csv", delimiter=',') # used to load data from text file.
    print(data[0])
    f1 = plt.figure(figsize=(10.0, 12.0), facecolor='khaki')    
    f1.suptitle("Curves from the data stored in the .csv file : ")
    a = f1.add_subplot(311) # it is used to add axes
    a.set_title("Sine Curve", color='red', fontsize=9, fontweight='bold')
    a.set_xlabel("Time")
    a.set_ylabel("sin(t)")
    a.grid(True)
    a.plot(data[0], data[1], color='red', linestyle='dashed')
    b = f1.add_subplot(313)
    b.set_title("Cosine Curve", color='green', fontsize=9, fontweight='bold')
    b.set_xlabel("Time")
    b.set_ylabel("cos(t)")
    b.grid(True)
    b.plot(data[0], data[2], color='green', linestyle='dashed')
openFile()
# Q8
plt.savefig("Figure2.png")
# Q9
window = tkinter.Tk()
window.title("Let's Play Wth Curves")
window.geometry("1000x900")
window.configure(bg="lightgreen")
def clicked():
    img = Image.open("Figure1.png")
    filename = ImageTk.PhotoImage(img)
    canvas = tkinter.Canvas(window, height=800, width=750)
    canvas.image = filename
    canvas.create_image(0, 0, anchor='nw', image=filename)
    canvas.pack()
but = tkinter.Button(window, text="Create Curve!", command=clicked, width=15, height=3, fg='red', bg='yellow', font="Arial")
# but.place(relx=0.5, rely=0.5, anchor=CENTER)
but.pack()
window.mainloop()