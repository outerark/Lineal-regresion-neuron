# Regresión Lineal con Descenso de Gradiente Animado 

Un proyecto educativo que implementa regresión lineal mediante descenso de gradiente, con visualización interactiva del proceso de entrenamiento.

![Ejemplo de ejecución]([https://drive.google.com/file/d/19hBLNnwgh-aKLRx2Z-SP30ZTFQiPbYSw/view?usp=sharing])  

## Características Clave 
- **Visualización interactiva** de la evolución de la recta de regresión.
- **Tasa de aprendizaje dinámica** que se ajusta automáticamente.
- Gráficos de error vs épocas/pendiente/punto de corte.
- Animación generada con `matplotlib.animation`.

## Parámetros Configurables

| Parámetro          | Tipo    | Descripción                                  | Valor por Defecto  |
|--------------------|---------|----------------------------------------------|--------------------|
| `m inicial`        | float   | Pendiente inicial de la recta                | 0.0                |
| `c inicial`        | float   | Término independiente inicial                | 0.0                |
| `learning rate`    | float   | Tasa de aprendizaje inicial                  | 0.0001             |
| `épocas`           | int     | Número de iteraciones de entrenamiento       | 100                |
| `learning dinámico`| bool    | Ajuste automático de tasa de aprendizaje     | False (0)          |

## Uso del Programa

### 1. Preparar los datos
Crea un archivo CSV con tus datos en el formato:
```csv
x,y
1.0,2.1
2.0,3.9
3.0,5.8
```
### 2. Ejecución básica
```bash
python main.py
```

## Requisitos 
```bash
pip install numpy pandas matplotlib

