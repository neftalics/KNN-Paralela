# Paralelización de KNN con MPI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MPI](https://img.shields.io/badge/MPI-mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementación paralela del algoritmo **K-Nearest Neighbors (KNN)** utilizando **MPI (Message Passing Interface)** en Python. Este proyecto demuestra técnicas de paralelización para algoritmos de Machine Learning mediante el modelo SPMD (Single Program, Multiple Data).

## 👥 Autores

- **Fabián Alvarado Ramos**
- **Eduardo Miguel Salas Palacios**
- **Neftalí Calixto Rojas**

**Institución:** Proyecto Universitario de Computación Paralela

---

## 📋 Descripción del Proyecto

El algoritmo KNN tiene una complejidad computacional de O(N×M), donde N es el número de puntos de entrenamiento y M el número de puntos de prueba. Este proyecto paraleliza el cálculo de distancias distribuyendo los puntos de prueba entre múltiples procesos usando MPI.

### Características Principales

- ✅ Implementación secuencial de referencia
- ✅ Tres versiones incrementales (Beta 1, Beta 2, Final)
- ✅ Modelo maestro-trabajador con topología DAG
- ✅ Operaciones colectivas MPI: `scatter`, `bcast`, `gather`
- ✅ Análisis completo de rendimiento (Speedup, Eficiencia, FLOPs)
- ✅ Escalabilidad con tamaño variable del problema

---

## 📁 Estructura del Repositorio

```
Paralela-Proyecto/
├── src/                          # Código fuente
│   ├── knn_sequential.py         # Versión secuencial (baseline)
│   ├── knn_parallel_v1.py        # Beta 1: Comunicación básica
│   ├── knn_parallel_v2.py        # Beta 2: Cómputo distribuido
│   ├── knn_parallel_final.py     # Versión final optimizada
│   └── knn_parallel_synthetic.py # Versión con datos sintéticos
│
├── analysis/                     # Scripts de análisis
│   ├── run_experiments.py        # Experimentos con variación de p
│   ├── run_scaling_experiments.py# Experimentos con variación de N
│   ├── plot_results.py           # Gráficas (tema oscuro)
│   └── plot_results_white.py     # Gráficas (tema blanco)
│
├── docs/                         # Documentación
│   ├── reporte_proyecto.tex      # Reporte académico (LaTeX)
│   ├── presentacion_beamer.tex   # Presentación (Beamer)
│   ├── images/                   # Gráficas tema oscuro
│   └── images_report/            # Gráficas tema blanco
│
├── results_log.csv               # Resultados experimentales (p variable)
├── scaling_log.csv               # Resultados de escalabilidad (N variable)
└── README.md                     # Este archivo
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.8+**
- **MPI Implementation:**
  - Windows: [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
  - Linux/Mac: OpenMPI o MPICH

### Dependencias de Python

```bash
pip install mpi4py numpy scikit-learn matplotlib pandas
```

### Verificar Instalación de MPI

```bash
mpiexec --version
```

---

## 💻 Uso

### 1. Ejecución Secuencial (Baseline)

```bash
python src/knn_sequential.py
```

**Salida esperada:**
```
Accuracy (Sequential): 0.9833
Execution Time: 5.02 seconds
```

### 2. Ejecución Paralela

#### Con 4 procesos:
```bash
mpiexec -n 4 python src/knn_parallel_final.py
```

#### Con 8 procesos:
```bash
mpiexec -n 8 python src/knn_parallel_final.py
```

**Salida esperada (Rank 0):**
```
=== KNN Parallel (Final Version) ===
Processes: 4
Accuracy: 0.9833
Total Time: 2.09 seconds
Compute Time: 1.85 seconds
Communication Time: 0.24 seconds
Speedup: 2.40x
Efficiency: 60.0%
```

### 3. Ejecutar Experimentos Completos

#### Variación del número de procesos (p ∈ {1,2,4,8}):
```bash
python analysis/run_experiments.py
```

#### Variación del tamaño del dataset (N variable):
```bash
python analysis/run_scaling_experiments.py
```

### 4. Generar Gráficas

#### Gráficas con tema oscuro (presentación):
```bash
python analysis/plot_results.py
```

#### Gráficas con fondo blanco (reporte):
```bash
python analysis/plot_results_white.py
```

**Gráficas generadas:**
- `time_vs_processes.png` - Análisis de tiempos
- `speedup.png` - Aceleración vs ideal
- `efficiency.png` - Eficiencia del clúster
- `flops.png` - Rendimiento computacional
- `scalability_n.png` - Escalabilidad con N

---

## 📊 Resultados Principales

### Dataset Utilizado
- **Nombre:** Digits (scikit-learn)
- **Características:** Imágenes 8×8 píxeles (64 dimensiones)
- **Muestras totales:** 1797
- **Train/Test split:** 80/20 (1437 train, 360 test)
- **Clases:** 10 (dígitos 0-9)

### Métricas de Rendimiento

| Procesos | Tiempo (s) | Speedup | Eficiencia | FLOPs/s |
|----------|------------|---------|------------|---------|
| 1        | 5.02       | 1.00x   | 100%       | 19.8M   |
| 2        | 2.68       | 1.87x   | 94%        | 37.1M   |
| 4        | 2.09       | 2.40x   | 60%        | 47.6M   |
| 8        | 2.01       | 2.50x   | 31%        | 49.5M   |

### Hallazgos Clave

- ✅ **Precisión idéntica:** 0.9833 (secuencial vs paralelo)
- ✅ **Reducción de tiempo:** 60% (5.02s → 2.01s)
- ✅ **Configuración óptima:** 4 procesos (mejor balance speedup/eficiencia)
- ⚠️ **Saturación:** A partir de 8 procesos, el overhead de comunicación domina
- 📈 **Complejidad confirmada:** O(N²) verificada experimentalmente

---

## 🔬 Metodología

### Arquitectura Maestro-Trabajador

```
┌─────────┐
│  Main   │  (Rank 0: Scatter datos)
└────┬────┘
     │
  ┌──┴──┬──────┬──────┐
  │     │      │      │
┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐
│ T1│ │ T2│ │ T3│ │ T4│  (Workers: Cómputo local)
└─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  │     │      │      │
  └──┬──┴──────┴──────┘
     │
┌────▼────┐
│   End   │  (Rank 0: Gather resultados)
└─────────┘
```

### Operaciones MPI Utilizadas

1. **`MPI_Scatter`**: Distribuir X_test entre procesos
2. **`MPI_Bcast`**: Enviar X_train, y_train a todos
3. **`MPI_Gather`**: Recolectar predicciones parciales

### Cálculo de FLOPs

Distancia euclidiana: `d(x,y) = √(Σ(xᵢ-yᵢ)²)`

- **FLOPs por distancia:** 3d (d restas + d multiplicaciones + d sumas)
- **FLOPs totales:** M × N × 3d = 360 × 1437 × 192 ≈ **99.5 MFLOPs**

---

## 📖 Documentación

### Reporte Académico

El reporte completo en LaTeX se encuentra en `docs/reporte_proyecto.tex` e incluye:

- ✅ Introducción y justificación
- ✅ Modelo PRAM y topología DAG
- ✅ Desarrollo incremental (3 versiones beta)
- ✅ Análisis de complejidad teórica normalizada
- ✅ Resultados experimentales completos
- ✅ Derivación de FLOPs desde distancia euclidiana
- ✅ Análisis de escalabilidad
- ✅ Conclusiones y mejoras propuestas
- ✅ Bibliografía con impacto descrito

### Compilar el Reporte

```bash
cd docs
pdflatex reporte_proyecto.tex
```

---

## 🎯 Mejoras Futuras

1. **Optimización de Comunicación:** Usar comunicación punto a punto para datasets grandes
2. **Estructuras de Datos Avanzadas:** Implementar KD-trees o Ball Trees
3. **Paralelización Híbrida:** Combinar MPI + OpenMP
4. **Balanceo Dinámico:** Distribución adaptativa de carga
5. **GPU Acceleration:** Porting a CUDA/OpenCL

---

## 📚 Referencias

1. **MPI Forum**. "MPI: A Message-Passing Interface Standard". Version 4.0, 2021.
2. **Gropp, W., Lusk, E., & Skjellum, A.** "Using MPI: Portable Parallel Programming with the Message-Passing Interface". MIT Press, 2014.
3. **mpi4py Documentation**. https://mpi4py.readthedocs.io/
4. **Scikit-learn**. "K-Nearest Neighbors". https://scikit-learn.org/stable/modules/neighbors.html
5. **Pacheco, P.** "An Introduction to Parallel Programming". Morgan Kaufmann, 2011.

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## 🤝 Contribuciones

Este es un proyecto académico. Para preguntas o sugerencias, contactar a los autores.

---

## 📞 Contacto

**Repositorio:** https://github.com/neftalics/KNN-Paralela

---

**Última actualización:** Noviembre 2025
