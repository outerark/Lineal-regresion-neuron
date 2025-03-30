"""
Created on Fri Apr 19, 2024
# -Cómo funcionan as redes neuronáis? Unha introdución práctica
STEMBach 2023 - 2025
Universidade de Vigo - Colexio Maristas Ourense
@author: Ángel López Rey
@author: Román Godás Vázquez
@author: Óscar Penín Blanco
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
from matplotlib.animation import FuncAnimation

# Carga el fichero de datos de entrada. Devuelve array eje X y array eje Y 
def cargar_datos(x, y):
    nombreFichero = input("Introduce el nombre del fichero (default=`data.csv`): ") or 'data.csv'
    if not os.path.isfile(nombreFichero):
        print(f"El fichero '{nombreFichero}' no existe. Usando 'data.csv' por defecto.")
        nombreFichero = 'data.csv'
    puntos = pd.read_csv(nombreFichero)
    for i in range(len(puntos)):
        x = np.append(x, float(puntos.iloc[i, 0]))
        y = np.append(y, float(puntos.iloc[i, 1]))
    return x, y

# Visualiza diagrama de dispersión, recta y precepto 
def graficar_datos(x, y, m=None, c=None):
    if m is not None and c is not None:
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = m * x_line + c
        plt.plot(x_line, y_line, label='Recta predicción', color='red')
    plt.scatter(x, y, label='Datos', color='blue')
    plt.grid(True)
    plt.legend()
    plt.show()

# Devuelve la funcion de coste. MSE
def calcular_variables(m, c, x, y):
    pred_y = np.dot(m, x) + c
    error = np.sqrt(np.sum((pred_y - y) ** 2) / len(y))
    return error, pred_y

# Calcula las derivadas parciales de la función de coste con respecto a m y c
# Devuelve el valor actualizado de m y c con el descenso de gradiente
def actualizar_parametros(m, c, x, pred_y, y, learning):
    Dm = -2 / len(x) * np.sum(x * (y - pred_y))
    Dc = -2 / len(x) * np.sum(y - pred_y)
    m = m - learning * Dm
    c = c - learning * Dc
    return m, c

# Visualiza gráfica de error frente a parámetro nombre 
def grafica_errores(errores, variable, nombre):
    plt.plot(variable, errores, color='red')
    plt.xlabel(f'Valores de {nombre}')
    plt.ylabel('Error cuadrático medio')
    plt.title(f'Error cuadrático medio con respecto a {nombre}')
    plt.grid(True)
    plt.show()

# Actualiza el estado de los elementos del siguiente frame de la gráfica animada  
def update(frame):
    m, c, error, epoca = frame
    Y_pred_min = m * np.min(x) + c
    Y_pred_max = m * np.max(x) + c
    line.set_data([np.min(x), np.max(x)], [Y_pred_min, Y_pred_max])
    text.set_text(f'm = {m:.4f}\nc = {c:.4f}\nError = {error:.4f}\nEpoca = {epoca}')
    return line, text

# Configura el estado inicial de la gráfica animada
def init():
    line.set_data([], [])
    text.set_text('')
    return line, text

# Declaración de las variables
m = None
c = None
ideal = False
epoca_ideal = 0
learning_ajuste = 0
executed = False
x = np.array([])
y = np.array([])
listaM = []
listaC = []
errores = []
frames = []
error_anterior = 50

# Cargar datos y graficar
x, y = cargar_datos(x, y)
graficar_datos(x, y, m, c)

# Asignación de los valores iniciales
m = float(input("Introduce el valor de m (default=0): ") or 0)
c = float(input("Introduce el valor inicial de c (default=0): ") or 0)
learning = float(input("Introduce el valor de la tasa de aprendizaje inicial (default=0.0001): ") or 0.0001)
epocas = int(input("Introduce el número de épocas (default=100): ") or 100)
dinamica = bool(int(input("¿Quieres que el learning rate sea dinámico? (1/0 default=0): ") or 0))
lrMaximo = learning * 5
lrMinimo = learning / 10
factorWarmUp = 1.1
factorDecay = 0.9

ventana_estabilizacion = 10
contador_estabilizacion = 0

# Proceso iterativo aproximación a la recta ideal
for i in range(epocas):
    error, pred_y = calcular_variables(m, c, x, y)
    listaM.append(m)
    listaC.append(c)
    errores.append(error)

    if dinamica:
        if error_anterior is not None:
            if error > error_anterior:
                # Se incremente el learning Rate si el error empeora hasta alcanzar el valor máximo
                learning = min((learning * factorWarmUp), lrMaximo)    
            else:
                # Se decrementa el learning Rate si el error mejora hasta alcanzar el valor mínimo
                learning = max((learning * factorDecay), lrMinimo)
       
    frames.append((m, c, error, i))
    print(f"Época: {i+1} m: {m} c: {c} error: {error} learning: {learning}\n")

    m, c = actualizar_parametros(m, c, x, pred_y, y, learning)

    # Determinar la época ideal basada en la estabilización del error
    if i > 1 and abs(error - error_anterior) / error_anterior < 0.005:
        contador_estabilizacion += 1
        if contador_estabilizacion >= ventana_estabilizacion and not executed:
            epoca_ideal = i - ventana_estabilizacion
            executed = True
    else:
        contador_estabilizacion = 0

    error_anterior = error

# Resultados
print(f"\nLa ejecución ha finalizado con un error de: {errores[-1]}")
print(f"La recta aproximada es: y = {listaM[-1]}x + {listaC[-1]}")
if executed:
    print(f"\nLa época en la que se alcanzó un valor ideal es la {epoca_ideal}\n")
else:
    print("\nNo se alcanzó una época ideal basada en la estabilización del error.\n")

# Graficar errores
grafica_errores(errores, np.arange(1, epocas + 1), "las épocas")
grafica_errores(errores, listaM, "la pendiente")
grafica_errores(errores, listaC, "el punto de corte con el eje y")

# Animación
fig, ax = plt.subplots()
ax.scatter(x, y)
line, = ax.plot([], [], color='red')
text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='left', va='top')
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000)
plt.show()
