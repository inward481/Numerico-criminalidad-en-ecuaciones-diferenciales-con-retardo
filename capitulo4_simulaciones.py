
"""
Simulaciones numéricas para el modelo de criminalidad con y sin retardo.

El script genera figuras y una tabla de indicadores para el Capítulo 4.
Requisitos:
    pip install numpy scipy pandas matplotlib
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================================================
# 1. Parámetros y funciones del modelo
# ============================================================

@dataclass
class Params:
    r: float = 1.0
    K: float = 1.0
    a: float = 1.0
    sigma: float = 0.9
    alpha: float = 0.2
    b: float = 1.4
    gamma: float = 0.4
    ell_e: float = 0.7


def rho(N: float, p: Params) -> float:
    """Crecimiento logístico generalizado usado en las simulaciones."""
    return p.r * (1.0 - N / p.K)


def rhs_ode(t: float, y: np.ndarray, p: Params) -> np.ndarray:
    """Campo vectorial del modelo sin retardo."""
    N, C = y
    den = p.sigma + max(N, 0.0)
    dN = N * rho(N, p) - p.a * N * C / den + p.alpha * N * C
    dC = -(p.gamma + p.ell_e) * C + p.b * N * C / den
    return np.array([dN, dC], dtype=float)


def rhs_dde(t: float, y: np.ndarray, y_tau: np.ndarray, p: Params) -> np.ndarray:
    """Campo vectorial del modelo con retardo."""
    N, C = y
    N_tau, C_tau = y_tau
    den = p.sigma + max(N, 0.0)
    den_tau = p.sigma + max(N_tau, 0.0)

    dN = N * rho(N, p) - p.a * N * C / den + p.alpha * N * C
    dC = -(p.gamma + p.ell_e) * C + p.b * N_tau * C_tau / den_tau
    return np.array([dN, dC], dtype=float)


def numero_reproductivo(p: Params) -> float:
    """Indicador local de invasión criminal alrededor de E1=(K,0)."""
    return p.b * p.K / ((p.sigma + p.K) * (p.gamma + p.ell_e))


def equilibrio_E1(p: Params) -> np.ndarray:
    return np.array([p.K, 0.0], dtype=float)


def equilibrio_E2(p: Params):
    """
    Equilibrio interior para rho(N)=r(1-N/K), cuando existe y es positivo.
    """
    remocion = p.gamma + p.ell_e
    if p.b <= remocion:
        return None

    N_star = p.sigma * remocion / (p.b - remocion)
    den_C = p.a / (p.sigma + N_star) - p.alpha

    if N_star <= 0 or den_C <= 0:
        return None

    C_star = p.r * (1.0 - N_star / p.K) / den_C

    if C_star <= 0:
        return None

    return np.array([N_star, C_star], dtype=float)


# ============================================================
# 2. Solucionadores numéricos
# ============================================================

def resolver_sin_retardo(p: Params, y0=(0.6, 0.4), t_end=80.0, n_eval=4000):
    """
    Resuelve el modelo sin retardo con solve_ivp.
    """
    t_eval = np.linspace(0.0, t_end, n_eval)
    sol = solve_ivp(
        fun=lambda t, y: rhs_ode(t, y, p),
        t_span=(0.0, t_end),
        y0=np.array(y0, dtype=float),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    y = sol.y.T
    y = np.maximum(y, 0.0)
    return sol.t, y


def resolver_con_retardo(p: Params, tau: float, historia, t_end=80.0, dt=0.01):
    """
    Resuelve el modelo con retardo mediante un esquema de pasos con RK4
    e interpolación lineal para el estado retardado.

    Para tau=0 se llama al solucionador de EDO, de modo que ambos casos
    puedan compararse dentro del mismo flujo de trabajo.
    """
    if tau <= 0:
        return resolver_sin_retardo(p, y0=historia(0.0), t_end=t_end, n_eval=int(t_end / dt) + 1)

    dt = min(dt, tau / 8.0)
    n_steps = int(np.ceil(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    y = np.zeros((n_steps + 1, 2), dtype=float)
    y[0, :] = np.array(historia(0.0), dtype=float)

    def estado_interpolado(s: float, i_actual: int) -> np.ndarray:
        if s <= 0.0:
            return np.array(historia(s), dtype=float)

        if s >= t[i_actual]:
            return y[i_actual, :].copy()

        j = int(np.floor(s / dt))
        j = max(0, min(j, i_actual - 1))
        theta = (s - t[j]) / (t[j + 1] - t[j])
        return (1.0 - theta) * y[j, :] + theta * y[j + 1, :]

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :].copy()

        def f(ts, ys):
            y_tau = estado_interpolado(ts - tau, i)
            return rhs_dde(ts, ys, y_tau, p)

        k1 = f(ti, yi)
        k2 = f(ti + dt / 2.0, yi + dt * k1 / 2.0)
        k3 = f(ti + dt / 2.0, yi + dt * k2 / 2.0)
        k4 = f(ti + dt, yi + dt * k3)

        y[i + 1, :] = yi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y[i + 1, :] = np.maximum(y[i + 1, :], 0.0)

    mask = t <= t_end + 1e-12
    return t[mask], y[mask, :]


# ============================================================
# 3. Indicadores computacionales
# ============================================================

def calcular_indicadores(t, y, equilibrio, eps=0.05):
    """
    Calcula indicadores para comparar escenarios:
    distancia final, amplitud final de C, tiempo de entrada en una vecindad
    del equilibrio, máximo de C y mínimo de N.
    """
    equilibrio = np.array(equilibrio, dtype=float)
    dist = np.linalg.norm(y - equilibrio, axis=1)

    ventana_final = t >= 0.75 * t[-1]
    A_N = float(y[ventana_final, 0].max() - y[ventana_final, 0].min())
    A_C = float(y[ventana_final, 1].max() - y[ventana_final, 1].min())

    # Primer tiempo desde el cual la solución permanece dentro de la vecindad eps.
    max_futuro = np.maximum.accumulate(dist[::-1])[::-1]
    indices = np.where(max_futuro <= eps)[0]
    T_eps = float(t[indices[0]]) if len(indices) > 0 else np.nan

    return {
        "distancia_final": float(dist[-1]),
        "A_N": A_N,
        "A_C": A_C,
        "T_epsilon": T_eps,
        "N_final": float(y[-1, 0]),
        "C_final": float(y[-1, 1]),
        "C_max": float(y[:, 1].max()),
        "N_min": float(y[:, 0].min()),
    }


# ============================================================
# 4. Funciones para guardar figuras
# ============================================================

def guardar_temporal(t, y, filename, title):
    plt.figure(figsize=(7, 4.2))
    plt.plot(t, y[:, 0], label=r"$N(t)$")
    plt.plot(t, y[:, 1], label=r"$C(t)$")
    plt.xlabel("Tiempo")
    plt.ylabel("Tamaño de población")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def guardar_fase(t, y, equilibrio, filename, title):
    plt.figure(figsize=(5.2, 4.6))
    plt.plot(y[:, 0], y[:, 1], label="Trayectoria")
    plt.scatter([equilibrio[0]], [equilibrio[1]], marker="o", label="Equilibrio de referencia")
    plt.xlabel(r"$N(t)$")
    plt.ylabel(r"$C(t)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def guardar_comparacion_tau(resultados, componente, filename, title):
    idx = 0 if componente == "N" else 1
    plt.figure(figsize=(7, 4.2))
    for tau, (t, y) in resultados.items():
        plt.plot(t, y[:, idx], label=rf"$\tau={tau}$")
    plt.xlabel("Tiempo")
    plt.ylabel(rf"${componente}(t)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def guardar_fase_tau(resultados, equilibrio, filename, title):
    plt.figure(figsize=(5.5, 4.8))
    for tau, (t, y) in resultados.items():
        plt.plot(y[:, 0], y[:, 1], label=rf"$\tau={tau}$")
    plt.scatter([equilibrio[0]], [equilibrio[1]], marker="o", label="Equilibrio")
    plt.xlabel(r"$N(t)$")
    plt.ylabel(r"$C(t)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def guardar_distancia_tau(resultados, equilibrio, filename, title):
    plt.figure(figsize=(7, 4.2))
    for tau, (t, y) in resultados.items():
        d = np.linalg.norm(y - equilibrio, axis=1)
        plt.plot(t, d, label=rf"$\tau={tau}$")
    plt.xlabel("Tiempo")
    plt.ylabel(r"$d_E(t)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def guardar_barras_metricas(df, filename, title):
    plt.figure(figsize=(7, 4.2))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width / 2, df["A_C"], width, label=r"$A_C(\tau)$")
    plt.bar(x + width / 2, df["distancia_final"], width, label=r"$d_E(t_f)$")
    plt.xticks(x, [str(v) for v in df["tau"]])
    plt.xlabel(r"Retardo $\tau$")
    plt.ylabel("Valor del indicador")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# 5. Ejecución principal
# ============================================================

def main():
    out = Path("figuras_capitulo4")
    out.mkdir(exist_ok=True)

    # Escenario 1: intervención institucional fuerte.
    p_fuerte = Params(r=1.0, K=1.0, a=1.0, sigma=0.9, alpha=0.2,
                      b=1.4, gamma=0.4, ell_e=0.7)

    # Escenario 2: intervención débil o nula.
    p_debil = Params(r=1.0, K=1.0, a=1.0, sigma=0.9, alpha=0.2,
                     b=1.4, gamma=0.4, ell_e=0.05)

    # Escenario 3: coexistencia interior.
    p_coex = Params(r=1.0, K=1.0, a=1.2, sigma=0.9, alpha=0.2,
                    b=1.4, gamma=0.3, ell_e=0.2)

    print("Número reproductivo escenario fuerte:", numero_reproductivo(p_fuerte))
    print("Número reproductivo escenario débil:", numero_reproductivo(p_debil))
    print("Número reproductivo escenario coexistencia:", numero_reproductivo(p_coex))
    print("E2 escenario débil:", equilibrio_E2(p_debil))
    print("E2 escenario coexistencia:", equilibrio_E2(p_coex))

    # --------------------------------------------------------
    # Modelo sin retardo
    # --------------------------------------------------------
    t, y = resolver_sin_retardo(p_fuerte, y0=(0.5, 0.4), t_end=80)
    E1 = equilibrio_E1(p_fuerte)
    guardar_temporal(t, y, out / "fig_4_1_intervencion_fuerte_temporal.png",
                     "Escenario con intervención institucional fuerte")
    guardar_fase(t, y, E1, out / "fig_4_2_intervencion_fuerte_fase.png",
                 "Plano de fase: intervención institucional fuerte")

    t, y = resolver_sin_retardo(p_debil, y0=(0.7, 0.05), t_end=80)
    E2_debil = equilibrio_E2(p_debil)
    guardar_temporal(t, y, out / "fig_4_3_intervencion_debil_temporal.png",
                     "Escenario con intervención institucional débil")
    guardar_fase(t, y, E2_debil, out / "fig_4_4_intervencion_debil_fase.png",
                 "Plano de fase: intervención institucional débil")

    t, y = resolver_sin_retardo(p_coex, y0=(0.9, 0.2), t_end=100)
    E2 = equilibrio_E2(p_coex)
    guardar_temporal(t, y, out / "fig_4_5_coexistencia_temporal.png",
                     "Escenario de coexistencia entre ambas poblaciones")
    guardar_fase(t, y, E2, out / "fig_4_6_coexistencia_fase.png",
                 "Plano de fase: equilibrio interior de coexistencia")

    # --------------------------------------------------------
    # Modelo con retardo: convergencia a E1
    # --------------------------------------------------------
    taus_E1 = [0.0, 0.5, 1.0, 2.0, 4.0]
    historia_E1 = lambda s: np.array([0.5, 0.4], dtype=float)
    resultados_E1 = {}
    indicadores = []

    for tau in taus_E1:
        t, y = resolver_con_retardo(p_fuerte, tau=tau, historia=historia_E1, t_end=80, dt=0.01)
        resultados_E1[tau] = (t, y)
        m = calcular_indicadores(t, y, equilibrio_E1(p_fuerte), eps=0.05)
        m.update({"escenario": "E1", "tau": tau})
        indicadores.append(m)

    guardar_comparacion_tau(resultados_E1, "N", out / "fig_4_7_retardo_E1_N.png",
                            "Efecto del retardo sobre N(t) cerca de E1")
    guardar_comparacion_tau(resultados_E1, "C", out / "fig_4_8_retardo_E1_C.png",
                            "Efecto del retardo sobre C(t) cerca de E1")
    guardar_fase_tau(resultados_E1, equilibrio_E1(p_fuerte), out / "fig_4_9_retardo_E1_fase.png",
                     "Trayectorias con retardo hacia E1")

    # --------------------------------------------------------
    # Modelo con retardo: equilibrio interior E2
    # --------------------------------------------------------
    taus_E2 = [0.0, 0.25, 0.5, 1.0, 2.0]
    historia_E2 = lambda s: np.array([0.9, 0.2], dtype=float)
    resultados_E2 = {}

    for tau in taus_E2:
        t, y = resolver_con_retardo(p_coex, tau=tau, historia=historia_E2, t_end=100, dt=0.01)
        resultados_E2[tau] = (t, y)
        m = calcular_indicadores(t, y, E2, eps=0.05)
        m.update({"escenario": "E2", "tau": tau})
        indicadores.append(m)

    guardar_comparacion_tau(resultados_E2, "N", out / "fig_4_10_retardo_E2_N.png",
                            "Efecto del retardo sobre N(t) cerca de E2")
    guardar_comparacion_tau(resultados_E2, "C", out / "fig_4_11_retardo_E2_C.png",
                            "Efecto del retardo sobre C(t) cerca de E2")
    guardar_fase_tau(resultados_E2, E2, out / "fig_4_12_retardo_E2_fase.png",
                     "Trayectorias con retardo hacia E2")
    guardar_distancia_tau(resultados_E2, E2, out / "fig_4_13_distancia_E2.png",
                          "Distancia al equilibrio interior para distintos retardos")

    df = pd.DataFrame(indicadores)
    df.to_csv(out / "indicadores_capitulo4.csv", index=False)

    df_E2 = df[df["escenario"] == "E2"].copy()
    guardar_barras_metricas(df_E2, out / "fig_4_14_metricas_tau_E2.png",
                            "Indicadores numéricos para distintos valores de retardo")

    print("\nIndicadores generados:")
    print(df[["escenario", "tau", "distancia_final", "A_C", "T_epsilon", "C_max", "N_min"]])
    print(f"\nArchivos guardados en: {out.resolve()}")


if __name__ == "__main__":
    main()
