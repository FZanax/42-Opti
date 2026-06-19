import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# =============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS DESDE LOS ARCHIVOS CSV
# =============================================================================

# --- Conjuntos de Nodos (Base, Centros, Plantas) ---
nodos_df = pd.read_csv('nodos.csv').dropna(subset=['id_nodo'])
N = nodos_df['id_nodo'].astype(int).tolist()
C = nodos_df[nodos_df['tipo'] == 'C']['id_nodo'].astype(int).tolist()
B = nodos_df[nodos_df['tipo'] == 'B']['id_nodo'].astype(int).tolist()

# --- Vehículos ---
vehiculos_df = pd.read_csv('vehiculos.csv', sep=';').dropna(subset=['id'])
V = vehiculos_df['id'].astype(int).tolist()
CapV = dict(zip(vehiculos_df['id'].astype(int), vehiculos_df['capacidad_max_ton'].astype(float)))
CostoKm = dict(zip(vehiculos_df['id'].astype(int), vehiculos_df['diesel_clp'].astype(float)))

# --- Plantas / Basurales ---
basurales_df = pd.read_csv('basurales.csv', sep=';').dropna(subset=['id'])
CapB = dict(zip(basurales_df['id'].astype(int), basurales_df['CapB'].astype(float)))
CapVac = dict(zip(basurales_df['id'].astype(int), basurales_df['CapVac'].astype(float)))
CostoEmergencia = dict(zip(basurales_df['id'].astype(int), basurales_df['CostoEmergencia_(clp/ton)'].astype(float)))
I0_plantas = dict(zip(basurales_df['id'].astype(int), basurales_df['Inventario_inicial_plantas'].astype(float)))
Cvac = dict(zip(basurales_df['id'].astype(int), basurales_df['objeto_vaciado_(clp/ton)'].astype(float))) # 'objeto_vaciado' es Cvac

# --- Costos Fijos por Centro (Cargados desde puertos.csv) ---
puertos_df = pd.read_csv('puertos.csv', sep=';').dropna(subset=['id'])
Costo_fijo_centro = dict(zip(puertos_df['id'].astype(int), puertos_df['costo_fijo'].astype(float)))

# --- Distancias entre Arcos ---
arcos_df = pd.read_csv('arcos.csv', sep=';').dropna(subset=['origen', 'destino'])
Dist = dict(zip(zip(arcos_df['origen'].astype(int), arcos_df['destino'].astype(int)), arcos_df['distancia_km'].astype(float)))

# --- Parámetros Globales ---
param_df = pd.read_csv('parametros_globales.csv', sep=';').dropna(subset=['parametro'])
param_dict = dict(zip(param_df['parametro'].str.strip(), param_df['valor'].astype(float)))

Presupuesto = param_dict['Presupuesto']
T_Max = int(param_dict['Tmax'])
Pt_base = param_dict['Pt']
MinRiesgo = param_dict['MinRiesgo']

# --- Biomasa e Inventario Inicial de Centros ---
biomasa_df = pd.read_csv('biomasa_centro.csv', sep=';').dropna(subset=['id_centro'])
# Eliminar columnas fantasma vacías (Unnamed) si las hay
biomasa_df = biomasa_df.loc[:, ~biomasa_df.columns.str.contains('^Unnamed')]

# Extraer el horizonte de tiempo real (Días 1 al 84)
T_dias = [int(x.split('_')[1]) for x in biomasa_df['id_centro'] if x != 'dia_0']

# Extraer inventario inicial I0 de los centros desde la fila 'dia_0'
row_i0 = biomasa_df[biomasa_df['id_centro'] == 'dia_0'].iloc[0]
I0_centros = {int(centro): float(row_i0[centro]) for centro in biomasa_df.columns if centro != 'id_centro'}

# Construir diccionario anidado Bio[centro][dia]
Bio = {c: {} for c in C}
for _, row in biomasa_df.iterrows():
    dia_str = row['id_centro']
    if dia_str == 'dia_0':
        continue
    t = int(dia_str.split('_')[1])
    for c in C:
        Bio[c][t] = float(row[str(c)])


# =============================================================================
# 2. CONSTRUCCIÓN DEL MODELO MATEMÁTICO EN GUROBI
# =============================================================================
model = gp.Model("Mitigacion_Marea_Roja_Real")
model.setParam('TimeLimit', 30 * 60) # 30 minutos de tiempo límite

# --- VARIABLES DE DECISIÓN ---
X = model.addVars(V, N, N, T_dias, vtype=GRB.BINARY, name="X")
Y = model.addVars(C, T_dias, vtype=GRB.BINARY, name="Y")
Q = model.addVars(V, C, T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="Q")
D = model.addVars(V, B, T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="D")
I = model.addVars(C, [0] + T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="I")
I_B = model.addVars(B, [0] + T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="I_B")
W = model.addVars(B, T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="W")
U = model.addVars(V, N, T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="U")
E = model.addVars(B, T_dias, vtype=GRB.CONTINUOUS, lb=0.0, name="E")
# Nueva variable: 1 si quedó inventario en el centro 'i' el día 't', 0 si no.
Quedo_Biomasa = model.addVars(C, T_dias, vtype=GRB.BINARY, name="Quedo_Biomasa")
# ACTIVACION LIMPIEZA
Activo = model.addVars(C, T_dias, vtype=GRB.BINARY, name="Activo")

# --- COMPONENTES DE LA FUNCIÓN OBJETIVO ACTUALIZADA ---
objetivo_transporte = gp.quicksum(CostoKm[v] * Dist[i,j] * X[v,i,j,t] for v in V for i in N for j in N for t in T_dias)
objetivo_inventario = gp.quicksum(Pt_base * Quedo_Biomasa[i,t] for i in C for t in T_dias)
objetivo_fijo = gp.quicksum(Costo_fijo_centro[i] * X[v,j,i,t] for v in V for i in C for j in N for t in T_dias)
objetivo_emergencia = gp.quicksum(CostoEmergencia[b] * E[b, t] for b in B for t in T_dias)
objetivo_vaciado = gp.quicksum(Cvac[b] * W[b,t] for b in B for t in T_dias)

# --- NUEVA VARIABLE DE SOBREGIRO ---
# Esta variable permite exceder el presupuesto, pero se penalizará
Sobregiro = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Sobregiro")

# Agregamos el sobregiro penalizado (ej: multiplicador alto) a la función objetivo
Penalizacion_Sobregiro = 1000 
model.setObjective(objetivo_transporte + objetivo_inventario + objetivo_fijo + objetivo_emergencia + objetivo_vaciado + (Sobregiro * Penalizacion_Sobregiro), GRB.MINIMIZE)

# --- RESTRICCIONES ---

# 1. Conservación de flujo
model.addConstrs((gp.quicksum(X[v,i,k,t] for i in N) == gp.quicksum(X[v,k,j,t] for j in N)
                  for v in V for k in N for t in T_dias), name="Conservacion_Flujo")

# 2. Equilibrio de inventario en centros
model.addConstrs((I[i, 0] == I0_centros[i] for i in C), name="Inventario_Inicial_Centros")
model.addConstrs((I[i, t] == I[i, t-1] + Bio[i][t] - gp.quicksum(Q[v, i, t] for v in V)
                  for i in C for t in T_dias), name="Equilibrio_Inventario")

# binaria de remanentes de basura lol
for i in C:
    # Definimos el M_local (la máxima biomasa posible en ese centro)
    M_local = I0_centros[i] + sum(Bio[i][t] for t in T_dias)
    
    for t in T_dias:
        model.addConstr(
            I[i, t] <= M_local * Quedo_Biomasa[i, t],
            name=f"Activa_Indicador_Inventario_{i}_{t}"
        )

# 3. Capacidad de los vehículos
model.addConstrs((gp.quicksum(Q[v,i,t] for i in C) <= CapV[v] for v in V for t in T_dias), name="CapVehiculo")

# 4. Conservación de carga vehicular (Lo cargado debe ser igual a lo descargado en basurales)
model.addConstrs((gp.quicksum(Q[v,i,t] for i in C) == gp.quicksum(D[v,b,t] for b in B) for v in V for t in T_dias), name="ConservacionCarga")

# 5. Lógica de recolección: Solo se puede cargar si el vehículo visita el centro
model.addConstrs((Q[v,i,t] <= CapV[v] * gp.quicksum(X[v,i,j,t] for j in N) for v in V for i in C for t in T_dias), name="LogicaRecoleccion")

# 6. Activación de limpieza Y
model.addConstrs((gp.quicksum(X[v,j,i,t] for v in V for j in N) >= Y[i,t] for i in C for t in T_dias), name="ActivacionLimpieza")

# 7: Ventana Sanitaria (regla 4 dias)
T_Max = 4  

for i in C:
    for t in range(min(T_dias), max(T_dias) - T_Max + 2):

        sum_limpiezas = gp.quicksum(Y[i, tau] for tau in range(t, t + T_Max))
        model.addConstr(
            sum_limpiezas >= Activo[i, t],
            name=f"Ventana_Sanitaria_Condicional_{i}_{t}"
        )

# 8 reglas de revision de limpieza 
for i in C:
    # Una Big-M exclusiva para el centro 'i', mucho más pequeña
    M_local = I0_centros[i] + sum(Bio[i][t] for t in T_dias)
    
    for t in T_dias:
        inventario_anterior = I0_centros[i] if t == min(T_dias) else I[i, t-1]
        inventario_disponible = inventario_anterior + Bio[i][t]
        
        # Regla A
        model.addConstr(
            inventario_disponible - MinRiesgo <= M_local * Activo[i, t],
            name=f"Dispara_Alerta_{i}_{t}"
        )
        
        # --- REGLA C: Solo se permite limpiar (Y) si el centro está "Activo" ---
        model.addConstr(
            Y[i, t] <= Activo[i, t],
            name=f"Permiso_Limpieza_{i}_{t}"
        )

# 9. Balance de inventario en plantas con Desborde por Incineración (Restricción blanda)
for b in B:
    for t in T_dias:
        inventario_anterior = I0_plantas[b] if t == min(T_dias) else I_B[b, t-1]
        
        # Ecuación unificada: Inv_Actual = Inv_Anterior + Descargado - Vaciado - Exceso_Incinerado
        model.addConstr(
            I_B[b, t] == inventario_anterior + gp.quicksum(D[v, b, t] for v in V) - W[b, t] - E[b, t],
            name=f"BalancePlanta_Desborde_{b}_{t}"
        )

# 10. El inventario físico remanente que se traspasa al día siguiente NO puede superar la capacidad máxima
model.addConstrs((I_B[b, t] <= CapB[b] for b in B for t in T_dias), name="CapAlmacenamientoMax")

# 11. Capacidad diaria de vaciado / procesamiento en plantas
model.addConstrs((W[b,t] <= CapVac[b] for b in B for t in T_dias), name="CapVaciado")

# --- NUEVA RESTRICCIÓN 12 (Blanda, incluyendo TODO costo y multa) ---
model.addConstr(
    objetivo_transporte + objetivo_fijo + objetivo_vaciado + objetivo_inventario + objetivo_emergencia <= Presupuesto + Sobregiro, 
    name="LimitePresupuestoTotal"
)

# 13. Eliminación de subcircuitos (MTZ)
N_sub = list(C) + list(B)
for v in V:
    for i in N_sub:
        for j in N_sub:
            if i != j:
                for t in T_dias:
                    model.addConstr(U[v,i,t] - U[v,j,t] + len(N) * X[v,i,j,t] <= len(N) - 1, name=f"Subtour_{v}_{i}_{j}_{t}")

# 14. Máximo una salida desde el puerto base por vehículo al día
model.addConstrs((gp.quicksum(X[v,0,j,t] for j in N) <= 1 for v in V for t in T_dias), name="SalidaPuerto")

# 15. Arcos prohibidos de la red logistica (Estructura obligatoria)
model.addConstrs((X[v,i,j,t] == 0 for v in V for i in C for j in C for t in T_dias), name="Prohibido_C_C")
model.addConstrs((X[v,b,i,t] == 0 for v in V for b in B for i in C for t in T_dias), name="Prohibido_B_C")
model.addConstrs((X[v,0,b,t] == 0 for v in V for b in B for t in T_dias), name="Prohibido_0_B")
model.addConstrs((X[v,i,0,t] == 0 for v in V for i in C for t in T_dias), name="Prohibido_C_0")

# --- OPTIMIZACIÓN ---
model.optimize()

# =============================================================================
# REPORTE DIVIDIDO: RESUMEN EN ARCHIVO Y DETALLES EN CONSOLA
# =============================================================================
if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    # Cálculos reales post-optimización
    costo_transporte_val = sum(CostoKm[v] * Dist[i,j] * X[v,i,j,t].X for v in V for i in N for j in N for t in T_dias)
    costo_fijo_val = sum(Costo_fijo_centro[i] * X[v,j,i,t].X for v in V for i in C for j in N for t in T_dias)
    costo_vaciado_val = sum(Cvac[b] * W[b,t].X for b in B for t in T_dias)
    costo_emergencia_val = sum(CostoEmergencia[b] * E[b, t].X for b in B for t in T_dias)
    costo_inventario_val = sum(Pt_base * I[i,t].X for i in C for t in T_dias)
    
    # LÓGICA DE SUMAS
    gasto_operativo = costo_transporte_val + costo_fijo_val + costo_vaciado_val
    gasto_multas = costo_inventario_val + costo_emergencia_val
    gasto_total = gasto_operativo + gasto_multas
    sobregiro_val = Sobregiro.X
    
    # Recopilar listas de eventos críticos
    dias_incineracion = [(t, b, E[b,t].X) for t in T_dias for b in B if E[b,t].X > 0.1]
    dias_alerta = [(t, i) for t in T_dias for i in C if Activo[i,t].X > 0.5]
    viajes_totales = sum(1 for t in T_dias for v in V for i in C for j in N if X[v,j,i,t].X > 0.5)
    total_recolectado = sum(Q[v,i,t].X for t in T_dias for v in V for i in C)

    # ---------------------------------------------------------
    # 1. ESCRIBIR RESUMEN ESTADÍSTICO EN ARCHIVO TXT
    # ---------------------------------------------------------
    with open("resumen_estadistico.txt", "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(" 📊 RESUMEN ESTADÍSTICO DE LA OPERACIÓN LOGÍSTICA \n")
        f.write("============================================================\n\n")
        
        f.write("💰 [BALANCE FINANCIERO TOTAL]\n")
        f.write(f" ► Presupuesto Base Asignado : {Presupuesto:,.0f} CLP\n")
        f.write(f" ► Gasto Operativo           : {gasto_operativo:,.0f} CLP\n")
        f.write(f" ► Multas por Inventario (Pt): {costo_inventario_val:,.0f} CLP\n")
        f.write(f" ► Multas Incineración (E)   : {costo_emergencia_val:,.0f} CLP\n")
        f.write(f" ► GASTO TOTAL EVALUADO      : {gasto_total:,.0f} CLP\n\n")
        
        if sobregiro_val > 0.1:
            f.write(f" ⚠️ SOBREGIRO REQUERIDO      : {sobregiro_val:,.0f} CLP\n")
        else:
            f.write(f" ✅ Presupuesto Suficiente (Sobró: {Presupuesto - gasto_total:,.0f} CLP)\n")
        
        f.write("\n🚚 [MÉTRICAS OPERATIVAS]\n")
        f.write(f" ► Total de viajes a centros : {viajes_totales} viajes\n")
        f.write(f" ► Biomasa total recolectada : {total_recolectado:,.1f} Toneladas\n")
        f.write("============================================================\n")
        
    print("\n[ÉXITO] El resumen estadístico se guardó en el archivo 'resumen_estadistico.txt'.")

    # ---------------------------------------------------------
    # 2. IMPRIMIR DETALLES DIARIOS EN LA CONSOLA (CMD)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" 📊 RESUMEN ESTADÍSTICO DE LA OPERACIÓN LOGÍSTICA (CONSOLA)")
    print("="*60)

    print("\n💸 [REGISTRO DE DEUDA POR INVENTARIO (Quedo_Biomasa)]")
    
    # Recopilamos los días y centros donde la variable Quedo_Biomasa tomó valor 1
    # Usamos > 0.5 por las tolerancias de los decimales en Gurobi
    dias_con_deuda = [(t, i) for t in T_dias for i in C if Quedo_Biomasa[i,t].X > 0.5]

    if not dias_con_deuda:
        print(" ✅ Excelente: Ningún centro generó deuda por biomasa remanente.")
    else:
        deudas_por_dia = {}
        # Agrupamos por día para que se lea mejor en la consola
        for t, c in dias_con_deuda:
            deudas_por_dia.setdefault(t, []).append(c)
            
        for t in sorted(deudas_por_dia.keys()):
            centros_str = ", ".join(map(str, deudas_por_dia[t]))
            print(f"  • Día {t:<3} | Se cobró deuda/multa en los Centros: {centros_str}")
    
    print("\n💰 [BALANCE FINANCIERO TOTAL]")
    print(f" ► Presupuesto Base Asignado : {Presupuesto:,.0f} CLP")
    print(f" ► Gasto Operativo           : {gasto_operativo:,.0f} CLP")
    print(f" ► Multas por Inventario (Pt): {costo_inventario_val:,.0f} CLP")
    print(f" ► Multas Incineración (E)   : {costo_emergencia_val:,.0f} CLP")
    print(f" ► GASTO TOTAL EVALUADO      : {gasto_total:,.0f} CLP\n")
    
    if sobregiro_val > 0.1:
        print(f" ⚠️ SOBREGIRO REQUERIDO      : {sobregiro_val:,.0f} CLP")
    else:
        print(f" ✅ Presupuesto Suficiente (Sobró: {Presupuesto - gasto_total:,.0f} CLP)")
    
    print("\n🚚 [MÉTRICAS OPERATIVAS]")
    print(f" ► Total de viajes a centros : {viajes_totales} viajes")
    print(f" ► Biomasa total recolectada : {total_recolectado:,.1f} Toneladas")

    print("\n🔥 [REGISTRO DE INCINERACIONES DE EMERGENCIA]")
    if not dias_incineracion:
        print(" ✅ Ninguna planta superó su capacidad.")
    else:
        for t, b, cantidad in dias_incineracion:
            print(f"  • Día {t:<3} | Planta {b}: Se incineraron {cantidad:.1f} Ton")

    print("\n🚨 [REGISTRO DE ALERTAS SANITARIAS ACTIVADAS]")
    if not dias_alerta:
        print(" ✅ Ningún centro superó el riesgo mínimo.")
    else:
        alertas_por_dia = {}
        for t, c in dias_alerta:
            alertas_por_dia.setdefault(t, []).append(c)
        for t in sorted(alertas_por_dia.keys()):
            centros_str = ", ".join(map(str, alertas_por_dia[t]))
            print(f"  • Día {t:<3} | Alerta en Centros: {centros_str}")
    
    print("\n🚚 [RUTAS DE VEHÍCULOS POR DÍA]")
    hay_viajes = False
    for t in T_dias:
        viajes_dia = []
        for v in V:
            for i in N:
                for j in N:
                    if X[v,i,j,t].X > 0.5:
                        # Identificar nombres claros para la consola
                        orig = f"Base 0" if i == 0 else (f"Centro {i}" if i in C else f"Planta {i}")
                        dest = f"Base 0" if j == 0 else (f"Centro {j}" if j in C else f"Planta {j}")
                        
                        viaje_str = f"  • Día {t:<3} | Camión {v:<2}: {orig} -> {dest}"
                        
                        # Agregar info de carga/descarga si corresponde
                        if i in C and Q[v,i,t].X > 0.01:
                            viaje_str += f" (Cargó {Q[v,i,t].X:.1f} Ton)"
                        if j in B and D[v,j,t].X > 0.01:
                            viaje_str += f" (Descargó {D[v,j,t].X:.1f} Ton)"
                            
                        viajes_dia.append(viaje_str)
                        
        if viajes_dia:
            hay_viajes = True
            for viaje in viajes_dia:
                print(viaje)
                
    if not hay_viajes:
        print("  Ningún camión registró movimientos.")

    print("="*60)

else:
    print("\n❌ El modelo no encontró una solución válida. Revisa los datos de entrada.")