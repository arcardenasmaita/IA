#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Criado: 10/04/2020
@Autor: ARCM 
@Tipo: Processo de Decisao de Markov MDP
@Algoritmo de Iteracao Valor (Value Iteration)
"""
import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import csv
import sys
import os
import matplotlib
import seaborn as sb
     
### Classe MDP

class MDP(object):
    def __init__(self, S, So, G, X, Y):        
        self.S = S   # S = numero de estados
        self.So = So # So = estado inicial
        self.G = G   # G = estado objetivo
        self.X = X   # X = eixo x
        self.Y = Y   # Y = eixo y
        self.A = 4   # nro de acoes posiveis        
        self.T_matrix = np.zeros([self.S, self.S, self.A]) 
        self.R_matrix = np.ones([self.S, self.A])
        self.R_matrix = self.R_matrix*(-1)
        self.R_matrix[self.S-1,:] = 0 
    def isEnd(self, state):        
        return state == self.G # es estado final?    
    def states(self):
        return range(0, self.S)
    def actions(self):
        return range(0, self.A) # 0:acima, 1:abaixo, 2:direita, 3:izquerda
    def R(self, state, accion):              
        return self.R_matrix[state][accion]
    def T(self, s_in, s_out, accion):
        T = np.zeros([self.S, self.S, self.A]) 
        #UP
        T[0][0][0] = 1
        T[1][1][0] = 1
        T[2][1][0] = 1
        T[3][3][0] = 1
        T[4][4][0] = 1
        T[5][0][0] = 1
        T[6][1][0] = 0.5
        T[6][5][0] = 0.5
        T[7][2][0] = 0.5
        T[7][5][0] = 0.5
        T[8][3][0] = 0.5
        T[8][5][0] = 0.5
        T[9][9][0] = 1
        
        #DOWN
        T[0][5][1] = 1
        T[1][6][1] = 1
        T[2][7][1] = 1
        T[3][8][1] = 1
        T[4][9][1] = 1
        T[5][5][1] = 1
        T[6][6][1] = 0.5
        T[6][5][1] = 0.5
        T[7][7][1] = 0.5
        T[7][5][1] = 0.5
        T[8][8][1] = 0.5
        T[8][5][1] = 0.5
        T[9][9][1] = 1
                  
        #RIGHT
        T[0][1][2] = 1
        T[1][2][2] = 1
        T[2][3][2] = 1
        T[3][4][2] = 1
        T[4][4][2] = 1
        T[5][6][2] = 1
        T[6][7][2] = 0.5
        T[6][5][2] = 0.5
        T[7][8][2] = 0.5
        T[7][5][2] = 0.5
        T[8][9][2] = 0.5
        T[8][5][2] = 0.5
        T[9][9][2] = 1
        
        
        #LEFT
        T[0][0][3] = 1
        T[1][0][3] = 1
        T[2][1][3] = 1
        T[3][2][3] = 1
        T[4][3][3] = 1
        T[5][5][3] = 1
        T[6][5][3] = 0.5
        T[6][6][3] = 0.5
        T[7][6][3] = 0.5
        T[7][7][3] = 0.5
        T[8][7][3] = 0.5
        T[8][8][3] = 0.5
        T[9][9][3] = 1
        
        self.T_matrix = T
        return T[s_in][s_out][accion]
    def printSolution(self, V, pi):
        # | 0 | 1 | 2 | 3 | 4 | y=0
        # | 5 | 6 | 7 | 8 | 9 | y=1
        #  x=0 x=1 x=2 x=3 x=4
        s_river = np.reshape(V,(self.X,self.Y))
        polices = np.reshape(pi,(self.X,self.Y))        
        print("Valores otimos")
        print(s_river)
        print("Politica ótima")
        print(polices)
        pi_grafico = np.empty(self.S, dtype='str')
        for p in range(self.S):
            if pi[p] == 0.:
                pi_grafico[p] = '^'
            elif pi[p] == 1.:
                pi_grafico[p] = 'v'
            elif pi[p] == 2.:
                pi_grafico[p] = '>'
            elif pi[p] == 3.:
                pi_grafico[p] = '<'
                
        for s in self.states():
            print('(S:{:2}, V:{:5}, pi:{:5} {:2})'.format(s, V[s], pi[s], pi_grafico[s]))
        
        Rm = Rm.drop([3], axis=1) 
        Rm = Rm.to_numpy()
        
        Vm = Rm[:,1]
        Pm = Rm[:,2]
        
        Vm = V.reshape(Y,X)
        #pi_grafico = Pm.reshape(Y,X)
        heat_map = sb.heatmap(Vm)
        plt.show()
        plt.savefig(os.path.join("Img_ambiente_%s_ep_%s_g_%s.png" % (ambiente,epsilon,ganma)), bbox_inches='tight')
          

### Algoritmos
        
def valueIteration(mdp, epsilon, ganma):
    V = np.zeros([mdp.S]) # almacenar o Voptimo[estado]    
    V_old = np.zeros([mdp.S])
    V[mdp.S-1] = 0.
    pi = np.zeros([mdp.S]) # politicas otimas
    res = np.inf
    
    Q = np.zeros([mdp.S,mdp.A])
    print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    while res > epsilon:
        np.copyto(V_old, V) 
        for state in mdp.states():
            for action in mdp.actions():
                Q[state,action] = mdp.R(state,action)
                for sNext in mdp.states():
                    Q[state,action] = Q[state,action] + ganma*mdp.T(state,sNext,action)*V_old[sNext]
                #print ('*{:1} {:15}'.format(state, Q[state,action]))
            V[state] = np.max(Q, axis=1)[state] 
            pi[state] = np.argmax(Q, axis=1)[state]
            Vm = V.reshape(mdp.X, mdp.Y)
            heat_map = sb.heatmap(Vm)
            plt.show()           
        # verificar a convergencia
        res = 0
        for s in mdp.states():
            dif = abs(V_old[s]-V[s])
            if dif > res:
                res = dif

    # escribir los resultados
    os.system('clear')
    
    #print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    for state in mdp.states():
        print ('{:1} {:15} {:15}'.format(state, V[state], pi[state]))
    return V,pi

mdp = MDP(S=10, So=5, G=9, X=2, Y=5)
V_opt, pi_opt = valueIteration(mdp, 0.000001, 0.8)
#mdp.printSolution(V_opt, pi_opt)