import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from tqdm import tqdm
from datetime import datetime
import seaborn as sns
from simulador_diabetes_rl import SimuladorDiabetesRL
from agente_q_learning import AgenteQLearning
import warnings
warnings.filterwarnings('ignore')
import os
import glob

class EvaluadorFinal:
    
    def __init__(self, modelo_path='Resultados/best_model/best_model.pkl', 
                 db_path='Base de datos/db_diabetes_50k.csv'):
        self.modelo_path = modelo_path
        self.db_path = db_path
        self.agente = None
        self.df_pacientes = None
        self.resultados = {}
    
    def cargar_mejor_modelo(self):

        
        # Crear directorio Resultados si no existe
        os.makedirs('Resultados', exist_ok=True)
        os.makedirs('Resultados/best_model', exist_ok=True)
        
        # Verificar si existe el archivo especificado
        if os.path.exists(self.modelo_path):
            print(f"1. Cargando modelo desde: {self.modelo_path}")
            with open(self.modelo_path, 'rb') as f:
                self.agente = pickle.load(f)
            print("   Modelo cargado exitosamente desde archivo .pkl")
        else:
            # Buscar el archivo más reciente en Resultados/best_model/
            print(f"1. Buscando mejor modelo disponible en Resultados/best_model/...")
            
            # Buscar archivos .pkl en Resultados/best_model/
            pkl_files = glob.glob("Resultados/best_model/*.pkl")
            
            if not pkl_files:
                print("   ERROR: No se encontraron archivos de modelo (.pkl)")
                return False
            
            # Ordenar por fecha de modificación (más reciente primero)
            pkl_files.sort(key=os.path.getmtime, reverse=True)
            mejor_modelo = pkl_files[0]
            
            print(f"Cargando: {mejor_modelo}")
            with open(mejor_modelo, 'rb') as f:
                self.agente = pickle.load(f)
            
            self.modelo_path = mejor_modelo
            print("Modelo cargado exitosamente")
        
        # Mostrar información del modelo cargado
        info = self.agente.get_info()
        print(f"2. Informacion del modelo:")
        print(f"  - Tamaño Q-table: {info['q_table_shape']}")
        print(f"  - Estados no cero: {info['q_table_non_zero']:,}")
        if 'metadata' in info:
            print(f"   - Fecha creacion: {info['metadata'].get('fecha_creacion', 'Desconocida')}")
            if 'mejor_puntuacion' in info['metadata']:
                print(f"   - Mejor puntuacion: {info['metadata']['mejor_puntuacion']:.1f}")
        
        return True
    
    def cargar_modelo_y_datos(self):

        # Cargar agente usando el método actualizado
        if not self.cargar_mejor_modelo():
            return False
        
        # Cargar pacientes
        print("\n3. Cargando base de datos de pacientes...")
        try:
            self.df_pacientes = pd.read_csv(self.db_path)
            print(f"   {len(self.df_pacientes):,} pacientes cargados")
            
            # Información de la base de datos
            print(f"\nInformacion del dataset:")
            print(f"  - Edad: {self.df_pacientes['edad'].min()}-{self.df_pacientes['edad'].max()} años")
            print(f"  - IMC: {self.df_pacientes['imc'].min():.1f}-{self.df_pacientes['imc'].max():.1f}")
            print(f"  - HbA1c: {self.df_pacientes['hba1c'].min():.1f}-{self.df_pacientes['hba1c'].max():.1f}%")
            
            # Distribución por tipo de diabetes
            tipos = self.df_pacientes['tipo_diabetes_especifico'].value_counts()
            for tipo, count in tipos.items():
                porcentaje = (count / len(self.df_pacientes)) * 100
                print(f"   - {tipo}: {count:,} ({porcentaje:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"   Error cargando datos: {e}")
            return False
    
    def crear_simulador_personalizado(self, paciente_data):
        # Usar factor_sensibilidad para determinar sensibilidad
        factor_sens = paciente_data.get('factor_sensibilidad', 50)
        
        if factor_sens < 40:
            sensibilidad = "baja"
        elif factor_sens < 60:
            sensibilidad = "normal"
        else:
            sensibilidad = "alta"
        
        # Crear simulador
        simulador = SimuladorDiabetesRL(
            paciente_id=paciente_data.get('id', 1),
            tipo_sensibilidad=sensibilidad
        )
        
        # Ajustar parámetros según paciente real
        peso = paciente_data.get('peso_kg', 70)
        simulador.metabolismo_basal *= (peso / 70.0)
        
        raciones = paciente_data.get('raciones_carbohidratos', 70)
        simulador.fuerza_comida = 25 * (raciones / 70.0)
        
        # Efecto de insulina personalizado
        simulador.efecto_insulina = 2.5 * (50 / max(20, factor_sens))
        
        # Variabilidad según años con diabetes
        años_diabetes = paciente_data.get('años_con_diabetes', 5)
        if años_diabetes > 10:
            simulador._variabilidad_extra = 3.0
        else:
            simulador._variabilidad_extra = 0.0
        
        return simulador
    
    def evaluar_muestra(self, n_pacientes=1000, n_episodios_por_paciente=3):
        print(f"\n4. Evaluando con {n_pacientes:,} pacientes ({n_episodios_por_paciente} episodios cada uno)...")
        
        if n_pacientes > len(self.df_pacientes):
            n_pacientes = len(self.df_pacientes)
            print(f"   Ajustando a {n_pacientes:,} pacientes (todos disponibles)")
        
        # Seleccionar muestra representativa
        muestra = self.df_pacientes.sample(n=n_pacientes, random_state=42)
        
        # Inicializar métricas
        metricas_totales = {
            'tiempo_en_rango': [],
            'hipoglucemias': [],
            'hiperglucemias': [],
            'recompensas': [],
            'glucosa_promedio': [],
            'acciones_promedio': [],
            'por_tipo_diabetes': {},
            'por_sensibilidad': {'baja': [], 'normal': [], 'alta': []},
            'por_edad': {
                'niños (<18)': [],
                'adultos (18-65)': [],
                'mayores (>65)': []
            }
        }
        
        # Barra de progreso
        pbar = tqdm(total=len(muestra), desc="Evaluando pacientes", unit="paciente")
        
        for _, paciente in muestra.iterrows():
            simulador = self.crear_simulador_personalizado(paciente.to_dict())
            
            # Determinar categorías del paciente
            tipo_diabetes = paciente['tipo_diabetes_especifico']
            edad = paciente['edad']
            factor_sens = paciente['factor_sensibilidad']
            
            # Clasificar por sensibilidad
            if factor_sens < 40:
                cat_sensibilidad = 'baja'
            elif factor_sens < 60:
                cat_sensibilidad = 'normal'
            else:
                cat_sensibilidad = 'alta'
            
            # Clasificar por edad
            if edad < 18:
                cat_edad = 'niños (<18)'
            elif edad <= 65:
                cat_edad = 'adultos (18-65)'
            else:
                cat_edad = 'mayores (>65)'
            
            # Inicializar categorías si no existen
            if tipo_diabetes not in metricas_totales['por_tipo_diabetes']:
                metricas_totales['por_tipo_diabetes'][tipo_diabetes] = {
                    'tiempo_en_rango': [],
                    'hipoglucemias': [],
                    'hiperglucemias': [],
                    'recompensas': [],
                    'count': 0
                }
            
            # Evaluar múltiples episodios por paciente
            for _ in range(n_episodios_por_paciente):
                estado = simulador.reset()
                estado_idx = self.agente.estado_a_indice(estado, simulador)
                terminado = False
                
                glucosas = []
                acciones = []
                recompensa_total = 0
                
                while not terminado:
                    # Solo explotación (sin exploración)
                    accion_idx = np.argmax(self.agente.q_table[estado_idx])
                    dosis = simulador.acciones[accion_idx]
                    acciones.append(dosis)
                    
                    nuevo_estado, recompensa, terminado = simulador.step(accion_idx)
                    nuevo_estado_idx = self.agente.estado_a_indice(nuevo_estado, simulador)
                    
                    estado_idx = nuevo_estado_idx
                    recompensa_total += recompensa
                    glucosas.append(simulador.glucosa)
                
                # Calcular métricas
                glucosas_array = np.array(glucosas)
                tiempo_rango = np.mean((glucosas_array >= 70) & (glucosas_array <= 180)) * 100
                hipoglucemias = np.mean(glucosas_array < 70) * 100
                hiperglucemias = np.mean(glucosas_array > 180) * 100
                glucosa_promedio = np.mean(glucosas_array)
                accion_promedio = np.mean(acciones)
                
                # Guardar métricas generales
                metricas_totales['tiempo_en_rango'].append(tiempo_rango)
                metricas_totales['hipoglucemias'].append(hipoglucemias)
                metricas_totales['hiperglucemias'].append(hiperglucemias)
                metricas_totales['recompensas'].append(recompensa_total)
                metricas_totales['glucosa_promedio'].append(glucosa_promedio)
                metricas_totales['acciones_promedio'].append(accion_promedio)
                
                # Guardar por tipo de diabetes
                metricas_totales['por_tipo_diabetes'][tipo_diabetes]['tiempo_en_rango'].append(tiempo_rango)
                metricas_totales['por_tipo_diabetes'][tipo_diabetes]['hipoglucemias'].append(hipoglucemias)
                metricas_totales['por_tipo_diabetes'][tipo_diabetes]['hiperglucemias'].append(hiperglucemias)
                metricas_totales['por_tipo_diabetes'][tipo_diabetes]['recompensas'].append(recompensa_total)
                metricas_totales['por_tipo_diabetes'][tipo_diabetes]['count'] += 1
                
                # Guardar por sensibilidad
                metricas_totales['por_sensibilidad'][cat_sensibilidad].append(tiempo_rango)
                
                # Guardar por edad
                metricas_totales['por_edad'][cat_edad].append(tiempo_rango)
            
            pbar.update(1)
        
        pbar.close()
        
        # Guardar resultados
        self.resultados = metricas_totales
        
        print(f"\nEvaluacion completada:")
        print(f"  - Total episodios: {len(metricas_totales['tiempo_en_rango']):,}")
        print(f"  - Pacientes evaluados: {len(muestra):,}")
        
        return metricas_totales
    
    def calcular_estadisticas(self):
        if not self.resultados:
            print("No hay resultados para analizar")
            return None
        
        print("RESULTADOS ESTADISTICOS FINALES\n")
        
        estadisticas = {}
        
        # Estadísticas generales
        print("\nESTADISTICAS GENERALES:")
        for key in ['tiempo_en_rango', 'hipoglucemias', 'hiperglucemias', 'recompensas', 'glucosa_promedio']:
            valores = self.resultados[key]
            if valores:
                mean = np.mean(valores)
                std = np.std(valores)
                min_val = np.min(valores)
                max_val = np.max(valores)
                
                print(f"  {key.replace('_', ' ').title():20s}: {mean:6.2f} ± {std:5.2f}  [ {min_val:6.2f} - {max_val:6.2f} ]")
                
                estadisticas[key] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'n': len(valores)
                }
        
        # Comparar con objetivos PDF
        tiempo_prom = estadisticas['tiempo_en_rango']['mean']
        hipo_prom = estadisticas['hipoglucemias']['mean']
        hiper_prom = estadisticas['hiperglucemias']['mean']
        
        print(f"\nCUMPLIMIENTO DE OBJETIVOS PDF:")
        print(f"  - Tiempo en rango >70%: {tiempo_prom:5.1f}% -> {'CUMPLE' if tiempo_prom > 70 else 'NO CUMPLE'}")
        print(f"  - Hipoglucemias <5%: {hipo_prom:5.1f}% -> {'CUMPLE' if hipo_prom < 5 else 'NO CUMPLE'}")
        print(f"  - Hiperglucemias <25%: {hiper_prom:5.1f}% -> {'CUMPLE' if hiper_prom < 25 else 'NO CUMPLE'}")
        
        # Por tipo de diabetes
        print(f"\nPOR TIPO DE DIABETES:")
        for tipo, data in self.resultados['por_tipo_diabetes'].items():
            if data['tiempo_en_rango']:
                mean = np.mean(data['tiempo_en_rango'])
                count = data['count']
                print(f"  - {tipo:30s}: {mean:5.1f}% tiempo en rango (n={count})")
        
        # Por sensibilidad
        print(f"\nPOR SENSIBILIDAD A INSULINA:")
        for sens, valores in self.resultados['por_sensibilidad'].items():
            if valores:
                mean = np.mean(valores)
                print(f"  - Sensibilidad {sens:10s}: {mean:5.1f}% tiempo en rango (n={len(valores)})")
        
        # Por edad
        print(f"\nPOR GRUPO DE EDAD:")
        for grupo, valores in self.resultados['por_edad'].items():
            if valores:
                mean = np.mean(valores)
                print(f"  - {grupo:20s}: {mean:5.1f}% tiempo en rango (n={len(valores)})")
        
        # Calcular puntuación compuesta
        puntuacion = (
            tiempo_prom * 0.5 + 
            (100 - hipo_prom * 10) * 0.3 + 
            (100 - hiper_prom * 2) * 0.2
        )
        print(f"\nPUNTUACION COMPUESTA: {puntuacion:.1f}/100")
        
        estadisticas['puntuacion_compuesta'] = puntuacion
        estadisticas['cumple_objetivos'] = {
            'tiempo_en_rango': tiempo_prom > 70,
            'hipoglucemias': hipo_prom < 5,
            'hiperglucemias': hiper_prom < 25
        }
        
        return estadisticas
    
    def visualizar_resultados(self, save_figures=True):
        if not self.resultados:
            print("No hay resultados para visualizar")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Resultados de Evaluacion Final del Modelo RL', fontsize=16, fontweight='bold')
        
        # 1. Histograma de tiempo en rango
        axes[0, 0].hist(self.resultados['tiempo_en_rango'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=70, color='r', linestyle='--', linewidth=2, label='Objetivo (70%)')
        axes[0, 0].axvline(x=np.mean(self.resultados['tiempo_en_rango']), 
                          color='g', linestyle='-', linewidth=2, label=f'Media ({np.mean(self.resultados["tiempo_en_rango"]):.1f}%)')
        axes[0, 0].set_title('Distribucion de Tiempo en Rango')
        axes[0, 0].set_xlabel('Tiempo en rango (70-180 mg/dL) [%]')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Boxplot por tipo de diabetes
        datos_tipo = []
        etiquetas_tipo = []
        for tipo, data in self.resultados['por_tipo_diabetes'].items():
            if data['tiempo_en_rango']:
                datos_tipo.append(data['tiempo_en_rango'])
                # Acortar etiquetas largas
                etiqueta = tipo.replace('diabetes_', '').replace('_', ' ').title()
                if len(etiqueta) > 20:
                    etiqueta = etiqueta[:17] + '...'
                etiquetas_tipo.append(etiqueta)
        
        if datos_tipo:
            bp = axes[0, 1].boxplot(datos_tipo, labels=etiquetas_tipo, patch_artist=True)
            # Colorear las cajas
            for patch in bp['boxes']:
                patch.set_facecolor(sns.color_palette()[0])
            axes[0, 1].axhline(y=70, color='r', linestyle='--', linewidth=2, alpha=0.7)
            axes[0, 1].set_title('Tiempo en Rango por Tipo de Diabetes')
            axes[0, 1].set_ylabel('Tiempo en rango [%]')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot: glucosa promedio vs tiempo en rango
        if len(self.resultados['glucosa_promedio']) > 0:
            scatter = axes[0, 2].scatter(self.resultados['glucosa_promedio'], 
                                        self.resultados['tiempo_en_rango'],
                                        c=self.resultados['hipoglucemias'], 
                                        cmap='RdYlGn_r', alpha=0.6, s=50)
            axes[0, 2].axvline(x=125, color='gray', linestyle=':', alpha=0.5, label='Ideal (125 mg/dL)')
            axes[0, 2].axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='Objetivo (70%)')
            axes[0, 2].set_title('Glucosa Promedio vs Tiempo en Rango')
            axes[0, 2].set_xlabel('Glucosa promedio [mg/dL]')
            axes[0, 2].set_ylabel('Tiempo en rango [%]')
            plt.colorbar(scatter, ax=axes[0, 2], label='Hipoglucemias [%]')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Comparativa de métricas clave
        metricas = ['tiempo_en_rango', 'hipoglucemias', 'hiperglucemias']
        valores = [np.mean(self.resultados[m]) for m in metricas]
        colores = ['green', 'red', 'orange']
        
        bars = axes[1, 0].bar(metricas, valores, color=colores, alpha=0.7)
        axes[1, 0].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Objetivo (70%)')
        axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Limite (5%)')
        axes[1, 0].axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Limite (25%)')
        
        # Añadir valores encima de las barras
        for bar, val in zip(bars, valores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom')
        
        axes[1, 0].set_title('Metricas Clinicas Principales')
        axes[1, 0].set_ylabel('Porcentaje [%]')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Distribución de acciones recomendadas
        if 'acciones_promedio' in self.resultados and self.resultados['acciones_promedio']:
            axes[1, 1].hist(self.resultados['acciones_promedio'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribucion de Dosis Promedio Recomendadas')
            axes[1, 1].set_xlabel('Dosis promedio [unidades]')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Heatmap de correlación
        datos_corr = pd.DataFrame({
            'Tiempo en rango': self.resultados['tiempo_en_rango'],
            'Hipoglucemias': self.resultados['hipoglucemias'],
            'Hiperglucemias': self.resultados['hiperglucemias'],
            'Glucosa promedio': self.resultados['glucosa_promedio'],
            'Recompensa': self.resultados['recompensas']
        })
        
        if len(datos_corr) > 1:
            corr_matrix = datos_corr.corr()
            im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 2].set_title('Matriz de Correlacion')
            axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 2].set_yticklabels(corr_matrix.columns)
            
            # Añadir valores de correlación
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                          ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
            
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Resultados/resultados_evaluacion_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nGraficos guardados como: {filename}")
        
        plt.show()
        
        return fig
    
    def guardar_resultados(self, nombre_base='resultados_evaluacion'):
        # Crear directorio Resultados si no existe
        os.makedirs('Resultados', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Guardar resultados detallados como JSON en Resultados/
        nombre_json = f"Resultados/{nombre_base}_{timestamp}.json"
        
        # Preparar datos para JSON (convertir numpy a tipos nativos de Python)
        resultados_json = {}
        for key, value in self.resultados.items():
            if key in ['por_tipo_diabetes', 'por_sensibilidad', 'por_edad']:
                # Estos son diccionarios anidados
                resultados_json[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        resultados_json[key][subkey] = [float(x) if isinstance(x, (np.floating, float)) else x for x in subvalue]
                    elif isinstance(subvalue, dict):
                        resultados_json[key][subkey] = {}
                        for k, v in subvalue.items():
                            if isinstance(v, list):
                                resultados_json[key][subkey][k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                            else:
                                resultados_json[key][subkey][k] = float(v) if isinstance(v, (np.floating, float)) else v
                    else:
                        resultados_json[key][subkey] = float(subvalue) if isinstance(subvalue, (np.floating, float)) else subvalue
            elif isinstance(value, list):
                resultados_json[key] = [float(x) if isinstance(x, (np.floating, float)) else x for x in value]
            else:
                resultados_json[key] = float(value) if isinstance(value, (np.floating, float)) else value
        
        # Añadir metadatos
        resultados_json['metadata'] = {
            'fecha_evaluacion': timestamp,
            'modelo_evaluado': self.modelo_path,
            'dataset': self.db_path,
            'n_pacientes_total': len(self.df_pacientes) if self.df_pacientes is not None else 0
        }
        
        with open(nombre_json, 'w', encoding='utf-8') as f:
            json.dump(resultados_json, f, indent=2, ensure_ascii=False)
        
        # 2. Guardar resumen ejecutivo como Markdown en Resultados/
        nombre_md = f"Resultados/{nombre_base}_resumen_{timestamp}.md"
        
        estadisticas = self.calcular_estadisticas()
        if estadisticas:
            with open(nombre_md, 'w', encoding='utf-8') as f:
                f.write("# Resumen de Evaluacion - Modelo RL para Diabetes\n\n")
                f.write(f"**Fecha de evaluacion:** {timestamp}\n\n")
                
                f.write("## Resultados Principales\n\n")
                f.write("| Metrica | Valor | Objetivo | Cumple |\n")
                f.write("|---------|-------|----------|--------|\n")
                
                objetivos = {
                    'tiempo_en_rango': ('>70%', estadisticas['tiempo_en_rango']['mean'] > 70),
                    'hipoglucemias': ('<5%', estadisticas['hipoglucemias']['mean'] < 5),
                    'hiperglucemias': ('<25%', estadisticas['hiperglucemias']['mean'] < 25)
                }
                
                for key, (objetivo, cumple) in objetivos.items():
                    valor = estadisticas[key]['mean']
                    f.write(f"| {key.replace('_', ' ').title()} | {valor:.1f}% | {objetivo} | {'SI' if cumple else 'NO'} |\n")
                
                f.write(f"\n**Puntuacion compuesta:** {estadisticas.get('puntuacion_compuesta', 0):.1f}/100\n\n")
                
                f.write("## Distribucion por Tipo de Diabetes\n\n")
                for tipo, data in self.resultados['por_tipo_diabetes'].items():
                    if data['tiempo_en_rango']:
                        mean = np.mean(data['tiempo_en_rango'])
                        f.write(f"- **{tipo.replace('_', ' ').title()}:** {mean:.1f}% tiempo en rango (n={data['count']})\n")
                
                f.write(f"\n## Archivos Generados\n")
                f.write(f"- `{nombre_json}`: Resultados completos en JSON\n")
                f.write(f"- `{nombre_md}`: Este resumen\n")
        
        print(f"\nRESULTADOS GUARDADOS:")
        print(f"   - {nombre_json} - Resultados completos")
        print(f"   - {nombre_md} - Resumen ejecutivo")
        
        return {
            'json': nombre_json,
            'md': nombre_md
        }
    
    def ejecutar_evaluacion_completa(self, n_pacientes=1000):
        # Cargar modelo y datos
        if not self.cargar_modelo_y_datos():
            return None
        
        # Evaluar muestra
        resultados = self.evaluar_muestra(n_pacientes=n_pacientes, n_episodios_por_paciente=3)
        
        # Calcular estadísticas
        estadisticas = self.calcular_estadisticas()
        
        # Visualizar resultados
        self.visualizar_resultados()
        
        # Guardar resultados
        archivos = self.guardar_resultados()
        
        # Análisis de política
        print(f"\n{'='*80}")
        print("ANALISIS DE POLITICA APRENDIDA")
        print(f"{'='*80}")
        
        # Crear un simulador de ejemplo para el análisis
        simulador_ejemplo = SimuladorDiabetesRL(paciente_id=999, tipo_sensibilidad="normal")
        
        # Verificar si el agente tiene método analizar_politica
        if hasattr(self.agente, 'analizar_politica'):
            self.agente.analizar_politica(simulador_ejemplo)
        else:
            print("El agente no tiene metodo 'analizar_politica'")
            # Mostrar información básica de la política
            print(f"\nInformacion basica de la politica:")
            print(f"- Acciones disponibles: {simulador_ejemplo.acciones}")
            print(f"- Tamaño Q-table: {self.agente.q_table.shape}")
            print(f"- Valores Q promedio: {np.mean(self.agente.q_table):.2f}")
        
        print(f"\n{'='*80}")
        print("EVALUACION COMPLETADA EXITOSAMENTE")
        print(f"{'='*80}")
        
        return {
            'resultados': resultados,
            'estadisticas': estadisticas,
            'archivos': archivos
        }

def main():

    print("TEST FINAL DEL MODELO RL - DIABETES TIPO 1")
    
    # Configuración - buscar en Resultados/best_model/
    MODELO_PATH = "Resultados/best_model/best_model.pkl"  # Ruta fija al mejor modelo
    DB_PATH = "Base de datos/db_diabetes_50k.csv"
    
    # Crear directorios necesarios
    os.makedirs('Resultados', exist_ok=True)
    os.makedirs('Resultados/best_model', exist_ok=True)
    
    # Verificar si existe la base de datos
    if not os.path.exists(DB_PATH):
        print(f"ERROR: No se encuentra la base de datos: {DB_PATH}")
        print("Asegurate de que el archivo existe en el directorio actual.")
        return
    
    try:
        df_temp = pd.read_csv(DB_PATH, nrows=1)
        total_pacientes = sum(1 for line in open(DB_PATH)) - 1  # Contar líneas menos el encabezado
        N_PACIENTES_EVALUAR = total_pacientes
    except:
        N_PACIENTES_EVALUAR = 1000
    
    print(f"\nConfiguracion:")
    print(f" - Modelo: {MODELO_PATH}")
    print(f" - Base de datos: {DB_PATH}")
    print(f" - Pacientes a evaluar: {N_PACIENTES_EVALUAR}")
    
    # Crear evaluador
    evaluador = EvaluadorFinal(modelo_path=MODELO_PATH, db_path=DB_PATH)
    
    # Ejecutar evaluación completa
    resultados = evaluador.ejecutar_evaluacion_completa(n_pacientes=N_PACIENTES_EVALUAR)
    
    if resultados:
        print(f"\nRESULTADO FINAL:")
        print(f"  - Pacientes evaluados: {N_PACIENTES_EVALUAR}")
        print(f"  - Modelo: {MODELO_PATH}")
        print(f"  - Dataset: {DB_PATH}")
        
        if 'estadisticas' in resultados and resultados['estadisticas']:
            stats = resultados['estadisticas']
            print(f"\nMETRICAS FINALES:")
            print(f"  - Tiempo en rango: {stats['tiempo_en_rango']['mean']:.1f}%")
            print(f"  - Hipoglucemias: {stats['hipoglucemias']['mean']:.1f}%")
            print(f"  - Hiperglucemias: {stats['hiperglucemias']['mean']:.1f}%")
            print(f"  - Puntuacion: {stats.get('puntuacion_compuesta', 0):.1f}/100")
        
        print(f"\nArchivos generados:")
        if 'archivos' in resultados:
            for tipo, archivo in resultados['archivos'].items():
                print(f"  - {archivo}")

    print("PROCESO FINALIZADO")

if __name__ == "__main__":
    main()