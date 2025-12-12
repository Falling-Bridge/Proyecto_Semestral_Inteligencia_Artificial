import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import json
from datetime import datetime
from simulador_diabetes_rl import SimuladorDiabetesRL
from agente_q_learning import AgenteQLearning

class EntrenadorInteligente:
    def __init__(self, db_path="db_diabetes_50k.csv"):
        self.db_path = db_path
        self.df_pacientes = None
        
        # USAR DIRECTAMENTE EL AGENTE IMPLEMENTADO
        self.agente = AgenteQLearning(n_estados=240, n_acciones=4)
        
        # Configuración de checkpoints
        self.checkpoint_interval = 500
        self.historial_checkpoints = []
        self.mejor_agente = None
        self.mejor_puntuacion = -float('inf')
        self.mejor_checkpoint_numero = 0
        
        # Hiperparámetros ajustables
        self.episodios_por_paciente = 3
        self.nombre_modelo_base = "best_model"
    
    def cargar_y_validar(self):

        print("\nCARGANDO Y VALIDANDO BASE DE DATOS...")
        
        try:
            self.df_pacientes = pd.read_csv(self.db_path)
            
            # Mezclar aleatoriamente
            self.df_pacientes = self.df_pacientes.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print("\n   DISTRIBUCIÓN DE PACIENTES:")
            print(f"   - Total pacientes: {len(self.df_pacientes):,}")
            print(f"   - Edad promedio: {self.df_pacientes['edad'].mean():.1f} años")
            print(f"   - IMC promedio: {self.df_pacientes['imc'].mean():.1f}")
            print(f"   - HbA1c promedio: {self.df_pacientes['hba1c'].mean():.1f}%")
            
            # Mostrar distribución por tipo
            if 'tipo_diabetes_especifico' in self.df_pacientes.columns:
                tipos = self.df_pacientes['tipo_diabetes_especifico'].value_counts()
                for tipo, count in tipos.items():
                    porcentaje = (count / len(self.df_pacientes)) * 100
                    print(f"   - {tipo}: {count:,} ({porcentaje:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def crear_simulador_personalizado(self, paciente_data):

        factor_sens = paciente_data.get('factor_sensibilidad', 50)
        if factor_sens < 40:
            sensibilidad = "baja"
        elif factor_sens < 60:
            sensibilidad = "normal"
        else:
            sensibilidad = "alta"
        
        simulador = SimuladorDiabetesRL(
            paciente_id=paciente_data.get('id', 1),
            tipo_sensibilidad=sensibilidad
        )
        
        # Ajustar parámetros según paciente
        peso = paciente_data.get('peso_kg', 70)
        simulador.metabolismo_basal *= (peso / 70.0)
        
        raciones = paciente_data.get('raciones_carbohidratos', 70)
        simulador.fuerza_comida = 25 * (raciones / 70.0)
        
        simulador.efecto_insulina = 2.5 * (50 / max(20, factor_sens))
        
        return simulador
    
    def evaluar_checkpoint(self, n_pacientes=100):
        print(f"\n   CHECKPOINT: Evaluando con {n_pacientes} pacientes...")
        
        if n_pacientes > len(self.df_pacientes):
            n_pacientes = len(self.df_pacientes)
        
        pacientes_eval = self.df_pacientes.sample(n=n_pacientes, random_state=int(datetime.now().timestamp()))
        
        metricas = {
            'tiempo_en_rango': [],
            'hipoglucemias': [],
            'hiperglucemias': [],
            'recompensas': [],
            'glucosa_promedio': []
        }
        
        for _, paciente in pacientes_eval.iterrows():
            simulador = self.crear_simulador_personalizado(paciente.to_dict())
            
            estado = simulador.reset()
            estado_idx = self.agente.estado_a_indice(estado, simulador)
            terminado = False
            
            glucosas = []
            recompensa_total = 0
            
            # Solo explotación durante evaluación
            while not terminado:
                accion = np.argmax(self.agente.q_table[estado_idx])
                nuevo_estado, recompensa, terminado = simulador.step(accion)
                nuevo_estado_idx = self.agente.estado_a_indice(nuevo_estado, simulador)
                
                estado_idx = nuevo_estado_idx
                recompensa_total += recompensa
                glucosas.append(simulador.glucosa)
            
            # Calcular métricas
            glucosas_array = np.array(glucosas)
            tiempo_rango = np.mean((glucosas_array >= 70) & (glucosas_array <= 180)) * 100
            hipoglucemias = np.mean(glucosas_array < 70) * 100
            hiperglucemias = np.mean(glucosas_array > 180) * 100
            
            metricas['tiempo_en_rango'].append(tiempo_rango)
            metricas['hipoglucemias'].append(hipoglucemias)
            metricas['hiperglucemias'].append(hiperglucemias)
            metricas['recompensas'].append(recompensa_total)
            metricas['glucosa_promedio'].append(np.mean(glucosas_array))
        
        # Calcular resultados
        resultados = {
            'tiempo_en_rango_prom': np.mean(metricas['tiempo_en_rango']),
            'hipoglucemias_prom': np.mean(metricas['hipoglucemias']),
            'hiperglucemias_prom': np.mean(metricas['hiperglucemias']),
            'recompensa_prom': np.mean(metricas['recompensas']),
            'glucosa_prom': np.mean(metricas['glucosa_promedio']),
            'pacientes_evaluados': n_pacientes,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Puntuación compuesta (ponderada)
        puntuacion = (
            resultados['tiempo_en_rango_prom'] * 0.5 +
            (100 - resultados['hipoglucemias_prom'] * 10) * 0.3 +
            (100 - resultados['hiperglucemias_prom'] * 2) * 0.2
        )
        
        resultados['puntuacion_compuesta'] = puntuacion
        
        return resultados
    
    def guardar_mejor_modelo(self, checkpoint_numero):
        if self.mejor_agente is not None:
            try:
                # Crear directorios
                os.makedirs('Resultados/best_model', exist_ok=True)
                
                # Actualizar agente con mejor Q-table
                self.agente.q_table = self.mejor_agente.copy()
                
                # Actualizar metadatos
                self.agente.metadata.update({
                    'mejor_checkpoint': checkpoint_numero,
                    'mejor_puntuacion': self.mejor_puntuacion,
                    'fecha_guardado': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'pacientes_procesados': checkpoint_numero * self.checkpoint_interval,
                    'tipo_entrenamiento': 'checkpoints_personalizados'
                })
                
                # Guardar usando el método del agente
                nombre_base_completo = f"Resultados/best_model/{self.nombre_modelo_base}"
                archivos = self.agente.guardar_modelo_completo(nombre_base_completo, usar_timestamp=False)
                
                print(f"   MEJOR MODELO GUARDADO:")
                print(f"      Checkpoint: {checkpoint_numero}")
                print(f"      Puntuación: {self.mejor_puntuacion:.1f}")
                print(f"      Ubicación: Resultados/best_model/")
                
                return True
                
            except Exception as e:
                print(f"   Error guardando mejor modelo: {e}")
                return False
        
        return False
    
    def entrenar_con_checkpoints(self, total_pacientes=None):
        if total_pacientes is None:
            total_pacientes = len(self.df_pacientes)
        
        print(f"\nINICIANDO ENTRENAMIENTO CON {total_pacientes:,} PACIENTES")
        print(f"   Checkpoints cada {self.checkpoint_interval} pacientes")
        print(f"   {self.episodios_por_paciente} episodios por paciente")
        
        # Crear estructura de carpetas
        os.makedirs('Resultados', exist_ok=True)
        os.makedirs('Resultados/checkpoints', exist_ok=True)
        os.makedirs('Resultados/best_model', exist_ok=True)
        
        pbar = tqdm(total=total_pacientes, desc="Pacientes procesados", unit="paciente")
        
        pacientes_procesados = 0
        checkpoints_completados = 0
        
        while pacientes_procesados < total_pacientes:
            pacientes_en_lote = min(self.checkpoint_interval, total_pacientes - pacientes_procesados)
            
            print(f"\nLOTE {checkpoints_completados + 1}: {pacientes_en_lote} pacientes")
            
            # Entrenar con el lote actual
            for i in range(pacientes_en_lote):
                idx = pacientes_procesados + i
                if idx >= len(self.df_pacientes):
                    break
                
                paciente = self.df_pacientes.iloc[idx]
                simulador = self.crear_simulador_personalizado(paciente.to_dict())
                
                # Entrenar episodios con este paciente
                for _ in range(self.episodios_por_paciente):
                    estado = simulador.reset()
                    estado_idx = self.agente.estado_a_indice(estado, simulador)
                    terminado = False
                    
                    while not terminado:
                        # USAR EL MÉTODO DEL AGENTE para selección de acción
                        accion = self.agente.seleccionar_accion(
                            estado_idx, 
                            episodio=pacientes_procesados, 
                            total_episodios=total_pacientes
                        )
                        
                        # Ejecutar acción
                        nuevo_estado, recompensa, terminado = simulador.step(accion)
                        nuevo_estado_idx = self.agente.estado_a_indice(nuevo_estado, simulador)
                        
                        # USAR EL MÉTODO DEL AGENTE para actualizar Q-table
                        self.agente.actualizar_q_table(
                            estado_idx, accion, recompensa, nuevo_estado_idx
                        )
                        
                        # Actualizar estado
                        estado_idx = nuevo_estado_idx
                
                pbar.update(1)
            
            pacientes_procesados += pacientes_en_lote
            checkpoints_completados += 1
            
            # Evaluar checkpoint
            resultados = self.evaluar_checkpoint(n_pacientes=min(200, len(self.df_pacientes)//10))
            
            # Guardar datos del checkpoint
            checkpoint_data = {
                'checkpoint_numero': checkpoints_completados,
                'pacientes_procesados': pacientes_procesados,
                'resultados': resultados,
                'q_table_mean': float(np.mean(self.agente.q_table)),
                'q_table_std': float(np.std(self.agente.q_table))
            }
            
            self.historial_checkpoints.append(checkpoint_data)
            
            # Mostrar resultados
            print(f"\n   RESULTADOS CHECKPOINT {checkpoints_completados}:")
            print(f"   - Pacientes totales: {pacientes_procesados:,}/{total_pacientes:,}")
            print(f"   - Tiempo en rango: {resultados['tiempo_en_rango_prom']:.1f}%")
            print(f"   - Hipoglucemias: {resultados['hipoglucemias_prom']:.1f}%")
            print(f"   - Hiperglucemias: {resultados['hiperglucemias_prom']:.1f}%")
            print(f"   - Glucosa promedio: {resultados['glucosa_prom']:.1f} mg/dL")
            print(f"   - Puntuación: {resultados['puntuacion_compuesta']:.1f}")
            
            # Actualizar mejor agente si corresponde
            if resultados['puntuacion_compuesta'] > self.mejor_puntuacion:
                self.mejor_puntuacion = resultados['puntuacion_compuesta']
                self.mejor_agente = self.agente.q_table.copy()
                self.mejor_checkpoint_numero = checkpoints_completados
                print(f"   NUEVO MEJOR AGENTE! Puntuación: {self.mejor_puntuacion:.1f}")
                
                # Guardar mejor modelo
                self.guardar_mejor_modelo(checkpoints_completados)
            
            # Guardar checkpoint individual
            self.guardar_checkpoint_individual(checkpoints_completados, resultados)
            
            # Ajustar hiperparámetros dinámicamente
            self.ajustar_hiperparametros(resultados)
        
        pbar.close()
        
        print("\nENTRENAMIENTO COMPLETADO")
        
        # Restaurar mejor agente al final
        if self.mejor_agente is not None:
            self.agente.q_table = self.mejor_agente
            print(f"Mejor agente restaurado (checkpoint {self.mejor_checkpoint_numero}, puntuación: {self.mejor_puntuacion:.1f})")
        
        return self.historial_checkpoints
    
    def ajustar_hiperparametros(self, resultados):
        # Ajustar exploración si hay muchas hipoglucemias
        if resultados['hipoglucemias_prom'] > 3:
            self.agente.epsilon = min(0.4, self.agente.epsilon * 1.1)
            print(f"   Ajuste: Aumentando exploración (ε={self.agente.epsilon:.2f})")
        
        # Ajustar tasa de aprendizaje si hay muchas hiperglucemias
        elif resultados['hiperglucemias_prom'] > 15:
            self.agente.alpha = min(0.4, self.agente.alpha * 1.1)
            print(f"   Ajuste: Aumentando tasa aprendizaje (α={self.agente.alpha:.2f})")
    
    def guardar_checkpoint_individual(self, checkpoint_numero, resultados):
        try:
            os.makedirs('Resultados/checkpoints', exist_ok=True)
            
            checkpoint_data = {
                'checkpoint_numero': checkpoint_numero,
                'pacientes_procesados': checkpoint_numero * self.checkpoint_interval,
                'q_table': self.agente.q_table.copy(),
                'resultados': resultados,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'hiperparametros': {
                    'alpha': self.agente.alpha,
                    'epsilon': self.agente.epsilon,
                    'gamma': self.agente.gamma
                }
            }
            
            nombre_archivo = f"Resultados/checkpoints/checkpoint_{checkpoint_numero:03d}.pkl"
            with open(nombre_archivo, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"   Checkpoint guardado: {nombre_archivo}")
            return nombre_archivo
            
        except Exception as e:
            print(f"   Error guardando checkpoint: {e}")
            return None
    
    def guardar_resultados(self, nombre_archivo="resultados_entrenamiento"):
        
        os.makedirs('Resultados', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar checkpoints
        checkpoint_file = f"Resultados/{nombre_archivo}_checkpoints_{timestamp}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.historial_checkpoints, f, indent=2, default=str)
        
        # Guardar modelo final
        model_file = f"Resultados/{nombre_archivo}_modelo_{timestamp}.npy"
        np.save(model_file, self.agente.q_table)
        
        # Guardar resumen
        resumen = {
            'total_pacientes': len(self.df_pacientes),
            'pacientes_entrenados': len(self.df_pacientes),
            'checkpoints_completados': len(self.historial_checkpoints),
            'mejor_puntuacion': self.mejor_puntuacion,
            'mejor_checkpoint': self.mejor_checkpoint_numero,
            'hiperparametros_finales': {
                'alpha': self.agente.alpha,
                'epsilon': self.agente.epsilon,
                'gamma': self.agente.gamma
            },
            'timestamp': timestamp
        }
        
        resumen_file = f"Resultados/{nombre_archivo}_resumen_{timestamp}.json"
        with open(resumen_file, 'w') as f:
            json.dump(resumen, f, indent=2)
        
        print(f"\nRESULTADOS GUARDADOS:")
        print(f"   - Checkpoints: {checkpoint_file}")
        print(f"   - Modelo: {model_file}")
        print(f"   - Resumen: {resumen_file}")
        
        return checkpoint_file, model_file, resumen_file
    
    def visualizar_progreso(self):
        if not self.historial_checkpoints:
            print("No hay datos de checkpoints para visualizar")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Progreso del Entrenamiento con Checkpoints', fontsize=14)
        
        checkpoints = [c['checkpoint_numero'] for c in self.historial_checkpoints]
        
        # 1. Tiempo en rango
        tiempo_rango = [c['resultados']['tiempo_en_rango_prom'] for c in self.historial_checkpoints]
        axes[0, 0].plot(checkpoints, tiempo_rango, 'g-', marker='o', linewidth=2)
        axes[0, 0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Objetivo (70%)')
        axes[0, 0].set_title('Tiempo en Rango (70-180 mg/dL)')
        axes[0, 0].set_xlabel('Checkpoint')
        axes[0, 0].set_ylabel('%')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Hipoglucemias
        hipoglucemias = [c['resultados']['hipoglucemias_prom'] for c in self.historial_checkpoints]
        axes[0, 1].plot(checkpoints, hipoglucemias, 'r-', marker='o', linewidth=2)
        axes[0, 1].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Límite seguro (5%)')
        axes[0, 1].set_title('Hipoglucemias (<70 mg/dL)')
        axes[0, 1].set_xlabel('Checkpoint')
        axes[0, 1].set_ylabel('%')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Hiperglucemias
        hiperglucemias = [c['resultados']['hiperglucemias_prom'] for c in self.historial_checkpoints]
        axes[0, 2].plot(checkpoints, hiperglucemias, 'orange', marker='o', linewidth=2)
        axes[0, 2].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Límite (25%)')
        axes[0, 2].set_title('Hiperglucemias (>180 mg/dL)')
        axes[0, 2].set_xlabel('Checkpoint')
        axes[0, 2].set_ylabel('%')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Puntuación compuesta
        puntuaciones = [c['resultados']['puntuacion_compuesta'] for c in self.historial_checkpoints]
        axes[1, 0].plot(checkpoints, puntuaciones, 'purple', marker='s', linewidth=2)
        best_idx = np.argmax(puntuaciones)
        axes[1, 0].plot(checkpoints[best_idx], puntuaciones[best_idx], 'gold', marker='*', markersize=15, label='Mejor')
        axes[1, 0].set_title('Puntuación Compuesta')
        axes[1, 0].set_xlabel('Checkpoint')
        axes[1, 0].set_ylabel('Puntuación')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Glucosa promedio
        glucosa_prom = [c['resultados']['glucosa_prom'] for c in self.historial_checkpoints]
        axes[1, 1].plot(checkpoints, glucosa_prom, 'blue', marker='o', linewidth=2)
        axes[1, 1].axhline(y=125, color='g', linestyle='--', alpha=0.5, label='Ideal (125 mg/dL)')
        axes[1, 1].set_title('Glucosa Promedio')
        axes[1, 1].set_xlabel('Checkpoint')
        axes[1, 1].set_ylabel('mg/dL')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Evolución de Q-values
        q_mean = [c['q_table_mean'] for c in self.historial_checkpoints]
        q_std = [c['q_table_std'] for c in self.historial_checkpoints]
        axes[1, 2].errorbar(checkpoints, q_mean, yerr=q_std, fmt='b-', marker='o', 
                           capsize=5, alpha=0.7, label='Media ± Desv')
        axes[1, 2].set_title('Evolución de Q-Values')
        axes[1, 2].set_xlabel('Checkpoint')
        axes[1, 2].set_ylabel('Valor Q')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'Resultados/progreso_entrenamiento_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    # Corregir la ruta con raw string
    entrenador = EntrenadorInteligente(db_path=r"Base de datos\db_diabetes_50k.csv")
    
    # Cargar y validar base de datos
    if not entrenador.cargar_y_validar():
        print("No se pudo cargar la base de datos. Verifica que db_diabetes_50k.csv exista.")
        return
    
    print("\nCONFIGURACIÓN DEL ENTRENAMIENTO:")
    print(f"   - Pacientes totales: {len(entrenador.df_pacientes):,}")
    print(f"   - Checkpoints cada: {entrenador.checkpoint_interval} pacientes")
    print(f"   - Episodios por paciente: {entrenador.episodios_por_paciente}")
    print(f"   - Tasa aprendizaje inicial: α={entrenador.agente.alpha}")
    print(f"   - Exploración inicial: ε={entrenador.agente.epsilon}")
    print(f"   - Factor descuento: γ={entrenador.agente.gamma}")
    
    input("\nPresiona Enter para comenzar el entrenamiento...")
    
    try:
        # Entrenar
        historial = entrenador.entrenar_con_checkpoints(
            total_pacientes=len(entrenador.df_pacientes)
        )
        
        # Guardar resultados
        entrenador.guardar_resultados()
        
        # Visualizar progreso
        entrenador.visualizar_progreso()
        
        # Evaluación final
        print("\nEVALUACIÓN FINAL EXHAUSTIVA")
        resultados_finales = entrenador.evaluar_checkpoint(n_pacientes=500)
        
        print(f"\nRESULTADOS FINALES (500 pacientes):")
        print(f"   Tiempo en rango: {resultados_finales['tiempo_en_rango_prom']:.1f}%")
        print(f"   Hipoglucemias: {resultados_finales['hipoglucemias_prom']:.1f}%")
        print(f"   Hiperglucemias: {resultados_finales['hiperglucemias_prom']:.1f}%")
        print(f"   Glucosa promedio: {resultados_finales['glucosa_prom']:.1f} mg/dL")
        print(f"   Puntuación final: {resultados_finales['puntuacion_compuesta']:.1f}")
        
        print(f"\nCUMPLIMIENTO DE OBJETIVOS:")
        print(f"   Tiempo en rango >70%: {resultados_finales['tiempo_en_rango_prom']:.1f}% {'(CUMPLE)' if resultados_finales['tiempo_en_rango_prom'] > 70 else '(NO CUMPLE)'}")
        print(f"   Hipoglucemias <5%: {resultados_finales['hipoglucemias_prom']:.1f}% {'(CUMPLE)' if resultados_finales['hipoglucemias_prom'] < 5 else '(NO CUMPLE)'}")
        print(f"   Hiperglucemias <25%: {resultados_finales['hiperglucemias_prom']:.1f}% {'(CUMPLE)' if resultados_finales['hiperglucemias_prom'] < 25 else '(NO CUMPLE)'}")
        
        print("\nENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
        print("Guardando resultados parciales...")
        entrenador.guardar_resultados(nombre_archivo="entrenamiento_interrumpido")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()