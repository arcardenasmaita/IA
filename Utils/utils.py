
import matplotlib.pyplot as plt
import numpy as np
import math

#---------------------------------------------------
#  GRAFICOS
#---------------------------------------------------
x = [[0.994, 0.864, 0.989, 0.733],
     [0.994, 0.930, 0.989, 0.733],
     [0.994, 0.921, 0.989, 0.733],
     [0.725, 0.924, 0.846, 0.730],
     [0.667, 0.813, 0.884, 0.714],
     [0.667, 0.813, 0.884, 0.714],
     [0.563, 0.497, 0.953, 0.793],
     [0.563, 0.497, 0.953, 0.793],
     [0.563, 0.497, 0.953, 0.793]]

def plot4graph(x, y1, y2, y3, y4, ):
    plt.style.use('ggplot')
    fuente={'fontname':'Tahoma'}
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    plt.subplot(411)
    plt.plot(x, y1, lw=2, color='#377eb8', label='Y')
    plt.legend()
    plt.ylabel('Y')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(412)
    plt.plot(x, y2, lw=2, color='#ff7f00', label='Y')
    plt.legend()
    plt.ylabel('Y')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(413)
    plt.plot(x, y3, lw=2, color='#4daf4a', label='Y')
    plt.legend()
    plt.ylabel('Y')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(414)
    plt.plot(x, y4, lw=2, color='#C82A54', label='Y4')
    plt.legend()
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.xscale('log')
    plt.yscale('log')

    plt.show()
    plt.savefig('img2.png', dpi=300)


#---------------------------------------------------
#  MATRICES
#---------------------------------------------------

#---------------------------------------------------
#  MAIN
#---------------------------------------------------
