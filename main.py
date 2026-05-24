import gurobipy as gp
from gurobipy import GRB
import scipy as scp
# -----------------------------------------------------------------------------
# SETUP
##################
'''[FALTA ESTO]'''
##################
# import N, C, B, V, T, Bio, CapV, CapB, Dist, CostoKm, Presupuesto, T_max, P, MinRiesgo, M
# DEFINI ESTO PARA QUE NO ME TIRE ERROR
N = [0, 1]
C = [0, 1]
B = [0, 1]
V = [0, 1]
T = [0, 1]
Bio = [0, 1]
CapV = [0, 1]
CapB = [0, 1]
Dist = [0, 1]
CostoKm = [0, 1]
Presupuesto = [0, 1]
T_Max = [0, 1]
P = [0, 1]
MinRiesgo = [0, 1]
M = [0, 1]
# NOMBRE
model = gp.Model("Mitigacion_Marea_Roja")
# TIME LIMIT 
model.setParam('TimeLimit', 30 * 60)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CONJUNTOS:
'''
N: Conjunto de todos los nodos del sistema, definido como N = {0} U C U B,
donde 0 representa el puerto base.
C ⊂ N: Subconjunto de centros de cultivo o lagos afectados por marea roja.
B ⊂ N: Subconjunto de plantas de tratamiento o basurales habilitados.
V : Conjunto de vehiculos disponibles (barcos y camiones), v ∈ {1, ..., V }.
T: Horizonte de planificacion en de las, t ∈ {1, ..., 7}.
'''

# -----------------------------------------------------------------------------
# VARIABLES/PARAMETROS:
'''
Bio_{i,t}: Nueva biomasa muerta (toneladas) generada en el centro i durante el dia t.
CapV_{v}: Capacidad maxima de carga del vehiculo v (toneladas).
CapB_{b}: Capacidad maxima de procesamiento diario de la planta/basural b.
Dist_{i,j} : Distancia en kilometros entre el nodo i y el nodo j.
CostoKm_{v}: Costo operativo por kilometro recorrido por el vehiculo v.
Presupuesto: Presupuesto total maximo asignado para la emergencia.
Tmax: Tiempo maximo permitido (en duas) que la biomasa puede estar en el agua antes de su descomposicion total (ej. 4 dias).
M: Numero constante positivo lo suficientemente grande (Big-M).
P: Ponderador (costo de penalizacion) por dejar biomasa en el agua.
MinRiesgo: Umbral minimo de generacion de biomasa por dia para que se considere riesgo de presencia de marea roja.
'''
# -----------------------------------------------------------------------------
# VARIABLES DE DECISION:

''' X_{v,i,j,t} ∈ {0, 1}: [(1) si el vehiculo v viaja desde el nodo i al nodo j en el dia t/ (0) en caso contrario] '''
# BINARIO
X = model.addVars(V, N, N, T, vtype=GRB.BINARY, name="X")

'''Y_{i,t} ∈ {0, 1}: [(1) si el centro i es visitado y limpiado en el dia t/ (0) en caso contrario]'''
# BINARIO
Y = model.addVars(C, T, vtype=GRB.BINARY, name="Y")

'''Q_{v,i,t} ≥ 0: Cantidad de biomasa (toneladas) cargada por el vehiculo v en el centro i en el dia t.'''
# CONTINUA
Q = model.addVars(V, C, T, vtype=GRB.CONTINUOUS, lb=0.0, name="Q")

'''D_{v,b,t} ≥ 0: Cantidad de biomasa (toneladas) descargada por el vehiculo v en la planta b en el dia t.'''
# CONTINUA
D = model.addVars(V, B, T, vtype=GRB.CONTINUOUS, lb=0.0, name="D")

'''I_{i,t} ≥ 0: Inventario de biomasa muerta acumulada en el centro i al final del dia t.'''
# CONTINUA
I = model.addVars(C, [0] + list(T), vtype=GRB.CONTINUOUS, lb=0.0, name="I")

'''U_{v,i,t} ≥ 0: Variable auxiliar continua que representa el orden o posicion de visita del nodo i por el vehiculo v en el dia t.'''
# CONTINUA
U = model.addVars(V, N, T, vtype=GRB.CONTINUOUS, lb=0.0, name="U")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# FUNCION OBJETIVO:
'''CORRECION AYUDANTE
Tenemos que encontrar una forma de que sea menos preferible que se recoja en los ultimos dias,
lo que se me ocurrio es que en la funcion objetivo se le agregue un multiplicador a la penalizacion
dependiendo de que dia se recoge la basura'''
objetivo_transporte = gp.quicksum(CostoKm[v] * Dist[i,j] * X[v,i,j,t] for v in V for i in N for j in N for t in T)
objetivo_inventario = gp.quicksum(P * I[i,t] for i in C for t in T)

model.setObjective(objetivo_transporte + objetivo_inventario, GRB.MINIMIZE)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# RESTRICCIONES:
##################
'''[REVISAR ESTO]'''
##################

# 1. Conservación de flujo de vehículos
'''Si un vehículo v entra al nodo k el día t, debe salir de él.'''
model.addConstrs((gp.quicksum(X[v, i, k, t] for i in N) == gp.quicksum(X[v, k, j, t] for j in N)
                  for v in V for k in N for t in T), name="Conservacion_Flujo_vehiculos")

# 2
'''CORRECION DEL AYUDANTE
En simples palabras, considerar que no se indefina en la incial
Para $t = 1$ (asumiendo inventario inicial cero o un parámetro conocido $I_{i,0}$):

$$I_{i,1} = Bio_{i,1} - \sum_{v\in V}Q_{v,i,1} \quad \forall i\in C$$

Para $t \in T \setminus \{1\}$ (o $t > 1$):

$$I_{i,t} = I_{i,t-1} + Bio_{i,t} - \sum_{v\in V}Q_{v,i,t} \quad \forall i\in C, \forall t \in T: t > 1$$'''

'''La biomasa al final del dia es igual a la del final del dia anterior, mas la nueva mortalidad
generada, menos lo extraido por todos los equipos '''
model.addConstrs((I[i,t] == (I[i,t-1] if t > 1 else 0) + Bio[i,t] - gp.quicksum(Q[v,i,t] for v in V) 
                  for i in C for t in T), name="Equilibrio_Inventario")

# 3. Capacidad máxima de los vehículos
'''Un vehículo no puede cargar más de lo que soporta.'''
model.addConstrs((gp.quicksum(Q[v, i, t] for i in C) <= CapV[v]
                  for v in V for t in T), name="Capacidad_Vehiculos")

# 4. Conservación de carga del vehículo
'''Todo lo que el vehiculo v recoge en los lagos, debe ser descargado en los basurales.'''
model.addConstrs((gp.quicksum(Q[v, i, t] for i in C) == gp.quicksum(D[v, b, t] for b in B)
                  for v in V for t in T), name="Conservacion_Carga")

# 5. Límite de capacidad de recepción en plantas/basurales
model.addConstrs((gp.quicksum(D[v, b, t] for v in V) <= CapB[b]
                  for b in B for t in T), name="Capacidad_Plantas")

# 6. Lógica de recolección - Relación entre ruta y carga
##################
'''CORRECION DEL AYUDANTE 
CAMBIAR BIG M por se redundante a CapV
$$Q_{v,i,t} \le CapV_{v} \cdot \sum_{j\in N}X_{v,i,j,t} \quad \forall v\in V, \forall i\in C, \forall t\in T$$'''
##################
'''Un equipo solo puede recolectar biomasa en un lugar si efectivamente viajo hacia alla'''
model.addConstrs((Q[v,i,t] <= CapV[v] * gp.quicksum(X[v,i,j,t] for j in N)
                  for v in V for i in C for t in T), name="Logica_Recoleccion")

# 7. Activación de limpieza 
'''Relaciona la variable binaria de visita con el ruteo de cualquier vehiculo hacia el centro.'''
model.addConstrs((gp.quicksum(X[v, i, j, t] for v in V for j in N) >= Y[i, t]
                  for i in C for t in T), name="Activacion_Limpieza")

# 8. Restriccion sanitaria cr´ıtica (Regla de los 4 d´ıas)
'''Todo centro de cultivo debe ser visitado y limpiado al menos una vez en cualquier ventana 
de Tmax dias consecutivos para evitar la pudricion total e hipoxia.'''
model.addConstrs((gp.quicksum(Y[i, tau] for tau in range(t, t + T_max)) >= 1 
                  for i in C for t in range(1, len(T) - T_max + 2)), name="Ventana_Sanitaria")

# 9. Límite de Presupuesto
model.addConstr(gp.quicksum(CostoKm[v] * Dist[i, j] * X[v, i, j, t] for v in V for i in N for j in N for t in T) <= Presupuesto,
                name="Limite_Presupuesto")

# 10. Eliminacion de subcircuitos
'''Garantiza que cada ruta sea una secuencia conectada que nazca en el puerto
base, impidiendo ciclos aislados de nodos.'''
N_sub = list(C) + list(B)
model.addConstrs((U[v, i, t] - U[v, j, t] + len(N) * X[v, i, j, t] <= len(N) - 1 
                  for v in V for i in N_sub for j in N_sub if i != j for t in T), name="Subtours")

# 11. Finalización de operaciones por umbral de riesgo 
##################
'''CORRECION DEL AYUDANTE 
Se agrega (I[i,t-1] if t > 1 else 0)
para que se pueda definir correctamente en el I[1,0] / que no sea I[1,-1]
PASAR ESTO A LATEX

Para el día 1:
$$Y_{i,1} \cdot MinRiesgo \le Bio_{i,1} \quad \forall i \in C$$
Para los días siguientes ($t > 1$):
$$Y_{i,t} \cdot MinRiesgo \le I_{i,t-1} + Bio_{i,t} \quad \forall i\in C, \forall t\in T: t > 1$$
'''
##################
'''En caso de que la cantidad de biomasa que se genere en una zona no alcance
el umbral minimo de riesgo de la marea roja, no se habilita la limpieza del
centro en ese dia.
'''
model.addConstrs((Y[i,t] * MinRiesgo <= (I[i,t-1] if t > 1 else 0) + Bio[i,t]
                  for i in C for t in T), name="Umbral_Marea_Roja")

# 12. Límite de salidas desde el puerto base
''' Se garantiza que cada vehiculo de la flota inicie su ruta 
saliendo del puerto base (nodo 0) a lo mas una vez por dia.'''
model.addConstrs((gp.quicksum(X[v, 0, j, t] for j in N) <= 1
                  for v in V for t in T), name="Salida_Puerto_Base")

# -----------------------------------------------------------------------------
