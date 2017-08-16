import tkinter
import numpy as np

window = tkinter.Tk()
a = 1.8
size = a*700
window.geometry("%dx%d" % (size,size))
canvas = tkinter.Canvas(window,width=size,height=size)
canvas.pack(fill=tkinter.BOTH, expand=1)

min_radius = a*20
max_radius = a*30
n = 3
tag = [1]
def add_circle(event):
    if tag[0] % 2== 1:
        n = 20
    else:
        n = 3
    tag[0] += 1
    canvas.delete('all')
    for i in range(n):
        for j in range(n):
            rad = size/n/3
            loc = (size/n/2 + i*size/n, size/n/2 + j*size/n)
            dist = (loc[0] - size/2)**2/1.6 + (loc[1] - size/2)**2
            if dist < (size/4)**2:
                c = np.random.randint(90,100)
                color = "#%02x%02x%02x"%((c,)*3)
                canvas.create_oval(loc[0] - rad,loc[1] - rad,loc[0] + rad, loc[1] + rad, fill=color)
                canvas.create_rectangle(i*size/n,j*size/n,(i+1)*size/n,(j+1)*size/n)


window.bind("<Button-1>",add_circle)
window.mainloop()
