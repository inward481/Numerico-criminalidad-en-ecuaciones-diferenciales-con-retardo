# Numerico-criminalidad-en-ecuaciones-diferenciales-con-retardo
Modelo dinámico de criminalidad mediante ecuaciones diferenciales con retardo, incorporando interacción saturante tipo Holling II, crecimiento logístico generalizado, intervención institucional y simulaciones numéricas para analizar estabilidad y efectos del tiempo de reacción.

# Modelo dinámico de criminalidad con retardo

Este repositorio contiene el código utilizado para simular y analizar un modelo dinámico de criminalidad basado en ecuaciones diferenciales ordinarias y ecuaciones diferenciales con retardo. El modelo describe la interacción entre una población no criminal \(N(t)\) y una población criminal \(C(t)\), incorporando crecimiento logístico generalizado, respuesta funcional saturante tipo Holling II, intervención institucional y tiempo de reacción.

## Descripción del modelo

El modelo estudia cómo la interacción social, la intervención institucional y el retardo temporal pueden modificar la evolución de las poblaciones. En particular, se analizan escenarios de:

- convergencia al equilibrio libre de criminalidad;
- persistencia de la población criminal;
- coexistencia entre población criminal y no criminal;
- oscilaciones inducidas por el retardo temporal.

El caso sin retardo permite estudiar la estabilidad mediante la matriz jacobiana y sus valores propios. El caso con retardo requiere analizar raíces características asociadas a una ecuación trascendental.

## Contenido del repositorio

```text
.
├── modelo_sin_retardo.py        # Simulaciones del sistema sin retardo
├── modelo_con_retardo.py        # Simulaciones del sistema con retardo
├── graficas/                    # Figuras generadas por las simulaciones
├── resultados/                  # Salidas numéricas e indicadores
└── README.md
