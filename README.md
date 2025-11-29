# Paralelización de KNN con MPI

Este proyecto implementa una versión paralela del algoritmo **K-Nearest Neighbors (KNN)** utilizando **MPI** (Message Passing Interface) en Python.

## Estructura del Proyecto

- `src/`: Código fuente (Secuencial, Beta 1, Beta 2, Final).
- `analysis/`: Scripts de análisis y generación de gráficas.
- `docs/`: Documentación, reporte en PDF y LaTeX.

## Requisitos

- Python 3.x
- MPI (MS-MPI en Windows, OpenMPI en Linux)
- `mpi4py`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`

## Ejecución

### Secuencial
```bash
python src/knn_sequential.py
```

### Paralelo (Ejemplo con 4 procesos)
```bash
mpiexec -n 4 python src/knn_parallel_final.py
```

## Resultados
Ver la carpeta `docs/` para el reporte completo.
