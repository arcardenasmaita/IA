#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creado el: 04/05/2020
@Actualizado el:
@Autor: ARCM 
@Fuente: https://www.youtube.com/watch?v=aIsgJJYrlXk&list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX&index=5
@Tipo: Problemas de búsqueda de la solucion
@Definicion:
 * Problema de transporte *
Bloques de calles numeradas de 1 a n
Acción 1: Caminar (walk) de 's' a 's+1' toma 1 minuto
Accion 2: Tomar un tren (tram) de 's' a '2s' toma 2 minutos
Cómo viajar de 1 a 'n' en el menor tiempo?

"""
import sys
sys.setrecursionlimit(10000)
### Modelo (problema de busqueda)

class TransportationProblem(object):
    def __init__(self, N):
        # N = numero de bloques de calles
        self.N = N
    def startState(self):
        # estado inicial
        return 1
    def isEnd(self, state):
        # estado final
        return state == self.N
    def succesorAndCost(self, state):
        # devuelve una lista de todas las posibles acciones en forma de la tripla (accion, nuevoEstado, costo)
        result = []
        if state+1 <= self.N:
            result.append(('walk', state+1, 1))
        if state*2 <= self.N:
            result.append(('tram', state*2, 2))
        return result

### Algoritmos
def printSolution(solution):
    totalCost, history = solution
    print('totalCost: {}'.format(totalCost))
    for item in history:
        print(item)

def backtrakingSearch(problem_def):
    # define la mejor solucion encotrada hasta aqui como un diccionario
    best = {
            'cost': float('+inf'), # costo inicial es infinito
            'history': None
            }
    def recurse(state, history, totalCost):
        # dado el 'state' donde se encuentra, habiendo experimentado
        # 'history' y acumulado un costo total de 'totalCost'
        # explore el resto del subarbol bajo el estado 'state'
        if problem_def.isEnd(state):
            # actualizar la mejor solución hasta aqui
            if totalCost < best['cost']:
                best['cost'] = totalCost
                best['history'] = history
            return
        # llamada recursiva a los hijos
        for action, newState, cost in problem_def.succesorAndCost(state):
            recurse(newState, history+[(action, newState, cost)], totalCost+cost)
    recurse(problem_def.startState(), history=[], totalCost=0)
    return (best['cost'], best['history'])
            
    
### Main

# ejemplo básico para 10 bloques
problem = TransportationProblem(N=1000)

# llamada simple para ver slo sucesores
print(problem.succesorAndCost(3))
print(problem.succesorAndCost(9))

# llamada al algoritmo de 'backtrakingSearch' pasando la definicion dem problema en 'problem'
printSolution(backtrakingSearch(problem))

 
