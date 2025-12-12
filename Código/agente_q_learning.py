import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time
import pickle
import pandas as pd
import json
import os
from datetime import datetime

class AgenteQLearning:
    
    def __init__(self, n_estados=240, n_acciones=4):
        """
        Args:
            n_estados: Número de estados (240 según propuesta)
            n_acciones: Número de acciones (4 según propuesta)
        """
        self.n_estados = n_estados
        self.n_acciones = n_acciones
        
        # Q-table inicializada con valores pequeños aleatorios para evitar estancamiento
        self.q_table = np.random.uniform(-1, 1, (n_estados, n_acciones)) * 0.1
        
        # Hiperparámetros
        self.alpha = 0.3      # Tasa de aprendizaje
        self.gamma = 0.95     # Factor de descuento
        self.epsilon = 0.3    # exploración inicial
        
        # Historial para monitoreo
        self.historial_recompensas = []
        self.historial_epsilon = []
        self.historial_exploracion = []
        
        # Metadatos del entrenamiento
        self.metadata = {
            'fecha_creacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': '1.0',
            'n_estados': n_estados,
            'n_acciones': n_acciones,
            'descripcion': 'Agente RL para control de diabetes tipo 1'
        }
    
    def seleccionar_accion(self, estado_idx, episodio=None, total_episodios=None):
        """
        Selecciona acción usando ε-greedy con ε decreciente
        """
        # ε decreciente
        if episodio is not None and total_episodios is not None:
            epsilon_actual = max(0.01, self.epsilon * (1 - (episodio / total_episodios) ** 0.8))
        else:
            epsilon_actual = self.epsilon
        
        # Guardar epsilon actual para historial
        self.historial_epsilon.append(epsilon_actual)
        
        # ε-greedy
        if random.random() < epsilon_actual:
            accion = random.randint(0, self.n_acciones - 1)
            explorando = True
        else:
            # Si hay empate, elegir aleatoriamente entre las mejores
            q_values = self.q_table[estado_idx]
            max_q = np.max(q_values)
            mejores_acciones = np.where(q_values == max_q)[0]
            accion = random.choice(mejores_acciones)
            explorando = False
        
        self.historial_exploracion.append(explorando)
        return accion
    
    def actualizar_q_table(self, estado_idx, accion_idx, recompensa, nuevo_estado_idx):
        """
        Actualiza Q-table usando la ecuación de Bellman
        """
        # Mejor Q-value del nuevo estado
        mejor_q_futuro = np.max(self.q_table[nuevo_estado_idx])
        
        # Q-value actual
        q_actual = self.q_table[estado_idx, accion_idx]
        
        # Ecuación de Bellman
        nuevo_q = q_actual + self.alpha * (recompensa + self.gamma * mejor_q_futuro - q_actual)
        
        # Actualizar Q-table
        self.q_table[estado_idx, accion_idx] = nuevo_q
    
    def estado_a_indice(self, estado, simulador):
        
        glucosa_cat, insulina_cat, tiempo_cat, sensibilidad_cat = estado
        
        try:
            # Mapear categorías a índices
            g_idx = simulador.categorias_glucosa.index(glucosa_cat)
            i_idx = simulador.categorias_insulina_activa.index(insulina_cat)
            t_idx = simulador.categorias_tiempo_dosis.index(tiempo_cat)
            s_idx = simulador.categorias_sensibilidad.index(sensibilidad_cat)
            
            # Índice único CORREGIDO: (g * 5 * 4 * 3) + (i * 4 * 3) + (t * 3) + s
            # Donde: 4 categorías glucosa, 5 categorías insulina, 4 categorías tiempo, 3 categorías sensibilidad
            # Total: 4 * 5 * 4 * 3 = 240 estados
            
            # Cálculo paso a paso:
            estados_por_sensibilidad = 1  # s_idx ya es el offset dentro del grupo
            estados_por_tiempo = 3  # 3 sensibilidades por cada categoría de tiempo
            estados_por_insulina = 4 * 3  # 4 tiempos * 3 sensibilidades por cada categoría de insulina
            
            indice = (g_idx * estados_por_insulina * 5) + (i_idx * estados_por_tiempo * 4) + (t_idx * 3) + s_idx
            
            # Verificar que esté en el rango correcto
            if indice >= self.n_estados:
                print(f"ADVERTENCIA: Índice {indice} fuera de rango. Ajustando...")
                indice = indice % self.n_estados
            
            return indice
            
        except Exception as e:
            print(f"Error convirtiendo estado a índice: {e}")
            print(f"Estado: {estado}")
            print(f"Categorías disponibles: glucosa={simulador.categorias_glucosa}")
            return 0  # Estado por defecto en caso de error
    
    def entrenar(self, simulador, n_episodios=1000):
        
        print(f"\n")
        print(f"ENTRENANDO AGENTE Q-LEARNING")
        print(f"Episodios: {n_episodios}")
        print(f"Estados: {self.n_estados}, Acciones: {self.n_acciones}")
        print(f"Hiperparámetros: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        print(f"")
        
        # Resetear historiales
        self.historial_recompensas = []
        self.historial_exploracion = []
        self.historial_epsilon = []
        
        # Estadísticas
        mejores_recompensas = deque(maxlen=100)
        tiempo_inicio = time.time()
        
        for episodio in range(n_episodios):
            # Estado inicial
            estado = simulador.reset()
            estado_idx = self.estado_a_indice(estado, simulador)
            terminado = False
            recompensa_total = 0
            pasos = 0
            
            while not terminado:
                # Seleccionar acción
                accion_idx = self.seleccionar_accion(estado_idx, episodio, n_episodios)
                
                # Ejecutar acción en el entorno
                nuevo_estado, recompensa, terminado = simulador.step(accion_idx)
                nuevo_estado_idx = self.estado_a_indice(nuevo_estado, simulador)
                
                # Actualizar Q-table
                self.actualizar_q_table(estado_idx, accion_idx, recompensa, nuevo_estado_idx)
                
                # Actualizar para siguiente iteración
                estado_idx = nuevo_estado_idx
                recompensa_total += recompensa
                pasos += 1
            
            # Registrar estadísticas
            self.historial_recompensas.append(recompensa_total)
            mejores_recompensas.append(recompensa_total)
            
            # Mostrar progreso
            if (episodio + 1) % 100 == 0:
                recompensa_promedio = np.mean(mejores_recompensas)
                exploracion_reciente = self.historial_exploracion[-pasos:] if pasos > 0 else [0]
                tasa_exploracion = np.mean(exploracion_reciente) * 100
                
                print(f"Episodio {episodio+1:4d}/{n_episodios} | "
                      f"Recompensa: {recompensa_total:6.1f} | "
                      f"Promedio (últimos 100): {recompensa_promedio:6.1f} | "
                      f"Exploración: {tasa_exploracion:5.1f}%")
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Actualizar metadatos
        self.metadata.update({
            'fecha_fin_entrenamiento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duracion_entrenamiento_seg': tiempo_total,
            'episodios_entrenados': n_episodios,
            'recompensa_promedio_final': float(np.mean(self.historial_recompensas[-100:])),
            'estados_visitados': int(np.count_nonzero(np.sum(np.abs(self.q_table), axis=1))),
            'estados_totales': self.n_estados,
            'hiperparametros': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        })
        
        print(f"\nENTRENAMIENTO COMPLETADO")
        print(f"Tiempo total: {tiempo_total:.1f} segundos")
        print(f"Recompensa promedio final: {np.mean(self.historial_recompensas[-100:]):.1f}")
        print(f"Estados visitados: {np.count_nonzero(np.sum(np.abs(self.q_table), axis=1))}/{self.n_estados}")
        
        return self.historial_recompensas
    
    def evaluar(self, simulador, n_episodios=10, verbose=True):

        print(f"\EVALUANDO AGENTE")
        print(f"Paciente ID: {simulador.paciente_id}")
        print(f"Sensibilidad: {simulador.tipo_sensibilidad}")
        
        metricas = {
            'recompensas': [],
            'tiempo_en_rango': [],
            'hipoglucemias': [],
            'hiperglucemias': [],
            'acciones_promedio': [],
            'glucosa_promedio': [],
            'glucosas_todas': []
        }
        
        for episodio in range(n_episodios):
            estado = simulador.reset()
            estado_idx = self.estado_a_indice(estado, simulador)
            terminado = False
            
            # Estadísticas del episodio
            glucosas = []
            acciones = []
            recompensa_total = 0
            
            while not terminado:
                # Solo explotación durante evaluación
                accion_idx = np.argmax(self.q_table[estado_idx])
                dosis = simulador.acciones[accion_idx]
                acciones.append(dosis)
                
                # Ejecutar acción
                nuevo_estado, recompensa, terminado = simulador.step(accion_idx)
                nuevo_estado_idx = self.estado_a_indice(nuevo_estado, simulador)
                
                # Actualizar
                estado_idx = nuevo_estado_idx
                recompensa_total += recompensa
                
                # Registrar glucosa
                info = simulador.get_info()
                glucosas.append(info['glucosa'])
            
            # Calcular métricas
            glucosas_array = np.array(glucosas)
            tiempo_rango = np.mean((glucosas_array >= 70) & (glucosas_array <= 180)) * 100
            hipoglucemias = np.mean(glucosas_array < 70) * 100
            hiperglucemias = np.mean(glucosas_array > 180) * 100
            accion_promedio = np.mean(acciones)
            glucosa_promedio = np.mean(glucosas_array)
            
            # Guardar métricas
            metricas['recompensas'].append(recompensa_total)
            metricas['tiempo_en_rango'].append(tiempo_rango)
            metricas['hipoglucemias'].append(hipoglucemias)
            metricas['hiperglucemias'].append(hiperglucemias)
            metricas['acciones_promedio'].append(accion_promedio)
            metricas['glucosa_promedio'].append(glucosa_promedio)
            metricas['glucosas_todas'].extend(glucosas)
            
            if verbose and n_episodios <= 20:
                print(f"Episodio {episodio+1}: "
                      f"Recompensa={recompensa_total:6.1f}, "
                      f"Tiempo en rango={tiempo_rango:5.1f}%, "
                      f"Hipo={hipoglucemias:4.1f}%, "
                      f"Hiper={hiperglucemias:4.1f}%, "
                      f"Glucosa promedio={glucosa_promedio:5.1f} mg/dL")
        
        # Calcular promedios
        print(f"\n")
        print(f"RESULTADOS FINALES (promedio de {n_episodios} episodios):")
        
        resultados = {}
        for key, values in metricas.items():
            if key != 'glucosas_todas' and values:
                promedio = np.mean(values)
                std = np.std(values) if len(values) > 1 else 0
                print(f"{key.replace('_', ' ').title():20s}: {promedio:6.2f} ± {std:5.2f}")
                resultados[key] = promedio
        
        # Comparar con objetivos del PDF
        tiempo_prom = np.mean(metricas['tiempo_en_rango'])
        hipo_prom = np.mean(metricas['hipoglucemias'])
        hiper_prom = np.mean(metricas['hiperglucemias'])
        
        print(f"\nVS OBJETIVOS PDF:")
        print(f"  Tiempo en rango >70%: {tiempo_prom:5.1f}% → {'ok' if tiempo_prom > 70 else 'wrong'}")
        print(f"  Hipoglucemias <5%: {hipo_prom:5.1f}% → {'ok' if hipo_prom < 5 else 'wrong'}")
        print(f"  Hiperglucemias <25%: {hiper_prom:5.1f}% → {'ok' if hiper_prom < 25 else 'wrong'}")
        
        return metricas, resultados
    
    # MÉTODOS DE GUARDADO
    
    def guardar_modelo_completo(self, nombre_base="modelo_qlearning", usar_timestamp=True):
        """
        Guarda el modelo completo en múltiples formatos.
        
        Args:
            nombre_base: Nombre base para los archivos
            usar_timestamp: Si True, añade timestamp a los nombres de archivo
        
        Returns:
            dict: Diccionario con las rutas de los archivos guardados
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if usar_timestamp:
            nombre_pkl = f"{nombre_base}_{timestamp}.pkl"
            nombre_json = f"{nombre_base}_metadata_{timestamp}.json"
            nombre_csv = f"{nombre_base}_qtable_{timestamp}.csv"
            nombre_npy = f"{nombre_base}_qtable_{timestamp}.npy"
        else:
            nombre_pkl = f"{nombre_base}.pkl"
            nombre_json = f"{nombre_base}_metadata.json"
            nombre_csv = f"{nombre_base}_qtable.csv"
            nombre_npy = f"{nombre_base}_qtable.npy"
        
        # Guardar objeto completo (pickle)
        with open(nombre_pkl, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Guardar metadata (JSON)
        with open(nombre_json, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Guardar Q-table (CSV)
        pd.DataFrame(self.q_table).to_csv(nombre_csv, index=False)
        
        # Guardar Q-table (NumPy)
        np.save(nombre_npy, self.q_table)
        
        return {
            'pkl': nombre_pkl,
            'json': nombre_json,
            'csv': nombre_csv,
            'npy': nombre_npy
        }
    
    def cargar_modelo_completo(self, archivo_pkl):

        try:
            with open(archivo_pkl, 'rb') as f:
                modelo_completo = pickle.load(f)
            
            # Cargar datos principales
            self.q_table = modelo_completo['q_table']
            
            # Cargar metadatos
            if 'metadata' in modelo_completo:
                self.metadata = modelo_completo['metadata']
            
            # Cargar hiperparámetros
            if 'hiperparametros' in modelo_completo:
                hiper = modelo_completo['hiperparametros']
                self.alpha = hiper.get('alpha', 0.3)
                self.gamma = hiper.get('gamma', 0.95)
                self.epsilon = hiper.get('epsilon', 0.3)
            
            # Cargar historiales
            if 'historial' in modelo_completo:
                historial = modelo_completo['historial']
                self.historial_recompensas = historial.get('recompensas', [])
                self.historial_exploracion = historial.get('exploracion', [])
                self.historial_epsilon = historial.get('epsilon', [])
            
            # Actualizar dimensiones
            self.n_estados, self.n_acciones = self.q_table.shape
            
            print(f"Modelo cargado exitosamente desde: {archivo_pkl}")
            print(f"  • Tamaño Q-table: {self.q_table.shape}")
            print(f"  • Fecha creación: {self.metadata.get('fecha_creacion', 'Desconocida')}")
            print(f"  • Estados no cero: {np.count_nonzero(self.q_table):,}")
            
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def cargar_desde_checkpoint(self, checkpoint_data):

        # checkpoint_data es un diccionario con 'q_table' y posiblemente más datos
        if 'q_table' in checkpoint_data:
            self.q_table = checkpoint_data['q_table']
            self.n_estados, self.n_acciones = self.q_table.shape
            
            # Actualizar metadatos
            self.metadata.update({
                'cargado_desde_checkpoint': True,
                'fecha_carga': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Modelo cargado desde checkpoint")
            print(f"  • Tamaño: {self.q_table.shape}")
            print(f"  • Estados no cero: {np.count_nonzero(self.q_table):,}")
            
            return True
        else:
            print("Checkpoint no contiene Q-table válida")
            return False
    
    # MÉTODOS DE ANÁLISIS Y VISUALIZACIÓN

    def analizar_politica(self, simulador):
        
        print(f"\n")
        print(f"ANÁLISIS DE POLÍTICA APRENDIDA")
        print(f"")
        
        # Estados clave para análisis
        estados_ejemplo = [
            # (glucosa, insulina_activa, tiempo_desde_dosis, sensibilidad)
            ("<70", 0, ">240", "normal"),      # Hipoglucemia
            ("70-180", 0, ">240", "normal"),   # Buen control
            ("180-250", 0, ">240", "normal"),  # Hiperglucemia leve
            (">250", 0, ">240", "normal"),     # Hiperglucemia severa
            ("70-180", 10, "0-60", "normal"),  # Insulina activa reciente
            ("70-180", 0, ">240", "baja"),     # Paciente resistente
            ("70-180", 0, ">240", "alta"),     # Paciente sensible
        ]
        
        print("\nPolítica para estados clave:")
        for estado in estados_ejemplo:
            estado_idx = self.estado_a_indice(estado, simulador)
            accion_recomendada = np.argmax(self.q_table[estado_idx])
            dosis = simulador.acciones[accion_recomendada]
            q_values = self.q_table[estado_idx]
            
            print(f"\nEstado: Glucosa={estado[0]}, Insulina activa={estado[1]}U")
            print(f"       Tiempo desde dosis={estado[2]}, Sensibilidad={estado[3]}")
            print(f"  → Acción recomendada: {dosis} unidades")
            print(f"  → Q-values: " + ", ".join([f"{q:.2f}" for q in q_values]))
        
        print(f"\nAnálisis de consistencia de la política:")
        
        # Verificar si la política es determinista
        decisiones = []
        for estado_idx in range(min(100, self.n_estados)):  # Muestra de 100 estados
            accion = np.argmax(self.q_table[estado_idx])
            decisiones.append(accion)
        
        unicas_acciones = set(decisiones)
        print(f" • Acciones únicas en muestra: {len(unicas_acciones)}/{self.n_acciones}")
        print(f" • Distribución de decisiones: {np.bincount(decisiones, minlength=self.n_acciones)}")
        
        # Calcular confianza promedio
        confianzas = []
        for estado_idx in range(min(100, self.n_estados)):
            q_values = self.q_table[estado_idx]
            if np.max(q_values) != np.min(q_values):  # Evitar división por cero
                confianza = (np.max(q_values) - np.min(q_values)) / (np.max(q_values) - np.min(q_values) + 1e-10)
                confianzas.append(confianza)
        
        if confianzas:
            print(f" • Confianza promedio: {np.mean(confianzas):.3f}")
        
        return estados_ejemplo
    
    def visualizar_aprendizaje(self, save_figure=True):
        
        if not self.historial_recompensas:
            print("No hay datos de entrenamiento para visualizar")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Proceso de Aprendizaje del Agente RL', fontsize=14, fontweight='bold')
        
        # 1. Recompensas por episodio
        axes[0, 0].plot(self.historial_recompensas, alpha=0.6, linewidth=0.5)
        axes[0, 0].set_title('Recompensa por Episodio')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Recompensa móvil
        ventana = min(100, len(self.historial_recompensas))
        recompensa_movil = [np.mean(self.historial_recompensas[max(0, i-ventana):i+1]) 
                           for i in range(len(self.historial_recompensas))]
        axes[0, 1].plot(recompensa_movil, 'r-', linewidth=2)
        axes[0, 1].set_title(f'Recompensa Promedio Móvil (ventana={ventana})')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Recompensa Promedio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Tasa de exploración
        if self.historial_exploracion:
            ventana_expl = min(1000, len(self.historial_exploracion))
            exploracion_movil = [np.mean(self.historial_exploracion[max(0, i-ventana_expl):i+1]) * 100
                                for i in range(len(self.historial_exploracion))]
            axes[0, 2].plot(exploracion_movil, 'g-', alpha=0.7, linewidth=1)
            axes[0, 2].set_title('Tasa de Exploración')
            axes[0, 2].set_xlabel('Paso')
            axes[0, 2].set_ylabel('Exploración (%)')
            axes[0, 2].set_ylim([0, 100])
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Epsilon a lo largo del tiempo
        if self.historial_epsilon:
            axes[1, 0].plot(self.historial_epsilon, 'b-', alpha=0.7, linewidth=1)
            axes[1, 0].set_title('Valor de ε (Exploración)')
            axes[1, 0].set_xlabel('Paso')
            axes[1, 0].set_ylabel('ε')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Distribución de Q-values
        q_values_flat = self.q_table.flatten()
        q_values_no_cero = q_values_flat[q_values_flat != 0]
        if len(q_values_no_cero) > 0:
            axes[1, 1].hist(q_values_no_cero, bins=50, alpha=0.7, color='purple')
            axes[1, 1].set_title(f'Distribución de Q-Values (≠0)')
            axes[1, 1].set_xlabel('Q-Value')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Añadir estadísticas al gráfico
            stats_text = f'Media: {np.mean(q_values_no_cero):.3f}\nStd: {np.std(q_values_no_cero):.3f}\nN: {len(q_values_no_cero):,}'
            axes[1, 1].text(0.95, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 6. Heatmap de Q-table (simplificado)
        if self.q_table.size <= 1000:  # Solo si no es muy grande
            im = axes[1, 2].imshow(self.q_table, aspect='auto', cmap='RdYlGn')
            axes[1, 2].set_title('Heatmap de Q-Table')
            axes[1, 2].set_xlabel('Acción')
            axes[1, 2].set_ylabel('Estado')
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_figure:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualizacion_aprendizaje_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado como: {filename}")
        
        plt.show()
        
        return fig
    
    def exportar_resumen(self, nombre_archivo='resumen_modelo.md'):

        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write("# Resumen del Modelo RL para Diabetes\n\n")
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuración\n")
            f.write(f"- Estados: {self.n_estados}\n")
            f.write(f"- Acciones: {self.n_acciones}\n")
            f.write(f"- α (tasa aprendizaje): {self.alpha}\n")
            f.write(f"- γ (factor descuento): {self.gamma}\n")
            f.write(f"- ε (exploración inicial): {self.epsilon}\n\n")
            
            f.write("## Estadísticas del Entrenamiento\n")
            if self.historial_recompensas:
                f.write(f"- Episodios entrenados: {len(self.historial_recompensas)}\n")
                f.write(f"- Recompensa promedio final: {np.mean(self.historial_recompensas[-100:]):.2f}\n")
                f.write(f"- Recompensa máxima: {np.max(self.historial_recompensas):.2f}\n")
                f.write(f"- Recompensa mínima: {np.min(self.historial_recompensas):.2f}\n\n")
            
            f.write("## Estadísticas de Q-Table\n")
            f.write(f"- Tamaño: {self.q_table.shape[0]} × {self.q_table.shape[1]}\n")
            f.write(f"- Celdas totales: {self.q_table.size}\n")
            f.write(f"- Celdas no cero: {np.count_nonzero(self.q_table)} ({(np.count_nonzero(self.q_table)/self.q_table.size*100):.1f}%)\n")
            f.write(f"- Valor medio: {np.mean(self.q_table):.4f}\n")
            f.write(f"- Desviación estándar: {np.std(self.q_table):.4f}\n")
            f.write(f"- Valor mínimo: {np.min(self.q_table):.4f}\n")
            f.write(f"- Valor máximo: {np.max(self.q_table):.4f}\n\n")
            
            f.write("## Distribución de Decisiones\n")
            # Calcular distribución de acciones óptimas
            acciones_optimas = [np.argmax(self.q_table[i]) for i in range(min(1000, self.n_estados))]
            distribucion = np.bincount(acciones_optimas, minlength=self.n_acciones)
            
            for i, count in enumerate(distribucion):
                porcentaje = (count / len(acciones_optimas)) * 100
                f.write(f"- Acción {i} ({self._get_accion_descripcion(i)}): {count} ({porcentaje:.1f}%)\n")
            
            f.write(f"\n## Metadatos\n")
            for key, value in self.metadata.items():
                f.write(f"- {key}: {value}\n")
        
        print(f"Resumen exportado como: {nombre_archivo}")
        return nombre_archivo
    
    def _get_accion_descripcion(self, accion_idx):
        """Obtiene descripción de la acción"""
        descripciones = {
            0: "0 unidades (no administrar)",
            1: "5 unidades",
            2: "10 unidades", 
            3: "15 unidades"
        }
        return descripciones.get(accion_idx, f"Acción {accion_idx}")
    
    def get_info(self):
        """Obtiene información resumida del agente"""
        return {
            'n_estados': self.n_estados,
            'n_acciones': self.n_acciones,
            
            'q_table_shape': self.q_table.shape,
            'q_table_non_zero': int(np.count_nonzero(self.q_table)),
            'q_table_mean': float(np.mean(self.q_table)),
            'q_table_std': float(np.std(self.q_table)),
            'hiperparametros': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            },
            'metadata': self.metadata
        }