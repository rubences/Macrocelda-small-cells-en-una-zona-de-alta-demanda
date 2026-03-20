# Macrocelda + Small Cells en Zona de Alta Demanda

Simulación Python de una red LTE heterogénea (HetNet) para una zona de alta demanda (calle comercial, festival o estadio perimetral). Compara el escenario de solo macrocelda frente a macro + dos small cells.

## Escenario

| Elemento | Parámetros |
|---|---|
| Macrocelda | 45 dBm · 25 m · 1800 MHz · 20 MHz BW |
| Small Cell 1 & 2 | 27 dBm · 5 m · 2600 MHz · 20 MHz BW |
| Área de simulación | 1 km × 1 km |
| Usuarios | 30–100 (con hotspots de alta densidad) |
| Modelo de propagación | COST-231 Hata |
| Scheduler | Round Robin y Proportional Fair |

## Estructura del proyecto

```
main.py              # Script principal de simulación
requirements.txt     # Dependencias Python
src/
  propagation.py     # Modelo COST-231 Hata (pérdida de trayectoria, RSRP)
  network.py         # BSs, UEs, selección de celda (CRE), SINR, throughput
  scheduler.py       # Scheduler Proportional Fair
  analysis.py        # Visualización: mapas RSRP, throughput, SINR, carga
tests/
  test_propagation.py
  test_network.py
  test_scheduler.py
  test_main.py
results/             # Figuras generadas (creada al ejecutar main.py)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar con 50 usuarios y parámetros por defecto
python main.py

# 100 usuarios con sesgo CRE de 6 dB para las small cells
python main.py --users 100 --cre 6

# Ver todas las opciones
python main.py --help
```

### Opciones de línea de comandos

| Opción | Por defecto | Descripción |
|---|---|---|
| `--users N` | 50 | Número de usuarios activos (30–100) |
| `--seed N` | 42 | Semilla aleatoria |
| `--cre dB` | 0 | Offset CRE para small cells (Cell Range Expansion) |
| `--pf-steps N` | 200 | Iteraciones del scheduler Proportional Fair |
| `--no-hotspot` | — | Distribución uniforme de usuarios (sin hotspots) |

## Resultados generados

Al ejecutar `main.py` se crean en `results/`:

- `rsrp_macro_only.png` – Mapa RSRP del escenario solo-macro
- `rsrp_hetnet.png` – Mapa RSRP del escenario HetNet
- `deployment_macro_only.png` – Emplazamientos y asociaciones (macro)
- `deployment_hetnet.png` – Emplazamientos y asociaciones (HetNet)
- `throughput_comparison.png` – CDF y box plot de throughput (comparativa)
- `sinr_histogram.png` – Distribución de SINR por escenario
- `load_distribution.png` – Carga de usuarios por celda (HetNet)

## Fórmulas implementadas

### Modelo de propagación COST-231 Hata
```
PL_dB = 46.3 + 33.9·log10(f) − 13.82·log10(h_tx) − a(h_rx)
        + (44.9 − 6.55·log10(h_tx))·log10(d) + C_dB
```

### Selección de celda con Cell Range Expansion (CRE)
```
Cell = argmax_i(RSRP_i + offset_bias_i)
```

### SINR
```
SINR = S / (N + I)
```

### Capacidad de Shannon
```
T_x = (BW / N_users) · log2(1 + SINR_x)
```

### Scheduler Proportional Fair
```
i*(t) = argmax_j [ R_j(k,t) / T_j(t) ]
```

## Tests

```bash
python -m pytest tests/ -v
```

## Dependencias open source

- **Python 3.8+**
- **NumPy** – cálculo numérico
- **Matplotlib** – visualización
- **SciPy** – funciones estadísticas

