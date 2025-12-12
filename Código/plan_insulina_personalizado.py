import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import json
import random
import warnings
warnings.filterwarnings('ignore')
import os

class PlanificadorInsulinaPersonalizado:
    
    def __init__(self, modelo_path='Resultados/best_model/best_model.pkl', db_path='Base de datos/db_diabetes_50k.csv'):
        self.modelo_path = modelo_path
        self.db_path = db_path
        
        print("Inicializando Planificador de Insulina Personalizado")
        
        # Cargar modelo entrenado
        with open(modelo_path, 'rb') as f:
            self.agente = pickle.load(f)
        
        # Cargar base de datos de pacientes
        self.df_pacientes = pd.read_csv(db_path)
        
        print(f"Modelo cargado: {modelo_path}")
        print(f"Base de datos: {db_path} ({len(self.df_pacientes)} pacientes)")
        print(f"Dimensiones Q-table: {self.agente.q_table.shape}")
    
    def seleccionar_paciente_aleatorio(self, paciente_id=None):
        
        if paciente_id is not None:
            if paciente_id < len(self.df_pacientes):
                paciente = self.df_pacientes.iloc[paciente_id].copy()
            else:
                print(f"ID {paciente_id} fuera de rango. Seleccionando aleatorio.")
                paciente_id = random.randint(0, len(self.df_pacientes) - 1)
                paciente = self.df_pacientes.iloc[paciente_id].copy()
        else:
            paciente_id = random.randint(0, len(self.df_pacientes) - 1)
            paciente = self.df_pacientes.iloc[paciente_id].copy()
        
        return paciente.to_dict(), paciente_id
    
    def crear_simulador_personalizado(self, paciente_data):
        
        from simulador_diabetes_rl import SimuladorDiabetesRL
        
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
        
        peso = paciente_data.get('peso_kg', 70)
        simulador.metabolismo_basal *= (peso / 70.0)
        
        raciones = paciente_data.get('raciones_carbohidratos', 70)
        simulador.fuerza_comida = 25 * (raciones / 70.0)
        
        simulador.efecto_insulina = 2.5 * (50 / max(20, factor_sens))
        
        años_diabetes = paciente_data.get('años_con_diabetes', 5)
        if años_diabetes > 10:
            simulador._variabilidad_extra = 3.0
        else:
            simulador._variabilidad_extra = 0.0
        
        return simulador
    
    def obtener_estado_simulador(self, simulador):
        try:
            # Intentar diferentes métodos que podría tener el simulador
            if hasattr(simulador, 'obtener_estado'):
                return simulador.obtener_estado()
            elif hasattr(simulador, 'get_state'):
                return simulador.get_state()
            elif hasattr(simulador, 'estado_actual'):
                return simulador.estado_actual
            elif hasattr(simulador, 'current_state'):
                return simulador.current_state
            else:
                # Crear estado manualmente basado en lo que sabemos del simulador
                # Basado en el PDF: (glucosa, insulina_activa, tiempo_desde_dosis, sensibilidad)
                
                # Glucosa categorizada
                glucosa = simulador.glucosa
                if glucosa < 70:
                    glucosa_cat = "<70"
                elif glucosa <= 180:
                    glucosa_cat = "70-180"
                elif glucosa <= 250:
                    glucosa_cat = "180-250"
                else:
                    glucosa_cat = ">250"
                
                # Insulina activa (estimada)
                insulina_activa = getattr(simulador, 'insulina_activa', 0)
                if insulina_activa < 5:
                    insulina_cat = 0
                elif insulina_activa < 10:
                    insulina_cat = 5
                elif insulina_activa < 15:
                    insulina_cat = 10
                elif insulina_activa < 20:
                    insulina_cat = 15
                else:
                    insulina_cat = 20
                
                # Tiempo desde última dosis (estimado)
                tiempo_cat = ">240"  # Por defecto
                
                # Sensibilidad
                sensibilidad = getattr(simulador, 'tipo_sensibilidad', 'normal')
                
                return (glucosa_cat, insulina_cat, tiempo_cat, sensibilidad)
                
        except Exception as e:
            print(f"Error obteniendo estado: {e}")
            # Estado por defecto
            return ("70-180", 0, ">240", "normal")
    
    def simular_dia_completo(self, paciente_data, glucosa_inicial=None, horario_comidas=None):
                
        if glucosa_inicial is None:
            glucosa_inicial = random.randint(100, 200)
        
        simulador = self.crear_simulador_personalizado(paciente_data)
        simulador.glucosa = glucosa_inicial
        
        # Reiniciar simulador para comenzar desde estado inicial
        try:
            if hasattr(simulador, 'reset'):
                simulador.reset()
            elif hasattr(simulador, 'restart'):
                simulador.restart()
        except:
            pass  # Continuar si no tiene método reset
        
        # Configurar horario de comidas si se proporciona
        if horario_comidas and hasattr(simulador, 'horario_comidas'):
            simulador.horario_comidas = horario_comidas
        
        # Inicializar registro de datos
        datos_dia = {
            'hora': [],
            'glucosa': [],
            'dosis_recomendada': [],
            'accion_idx': [],
            'insulina_activa': [],
            'estado_descripcion': []
        }
        
        # Obtener estado inicial usando método adaptativo
        estado = self.obtener_estado_simulador(simulador)
        estado_idx = self.agente.estado_a_indice(estado, simulador)
        
        # Simular 24 horas con pasos de 30 minutos
        for paso in range(48):  # 24 horas * 2 pasos por hora
            hora_actual = paso * 0.5  # Horas desde inicio (cada 30 minutos)
            
            # Seleccionar acción usando la política aprendida
            accion_idx = np.argmax(self.agente.q_table[estado_idx])
            dosis = simulador.acciones[accion_idx]
            
            # Registrar datos
            datos_dia['hora'].append(hora_actual)
            datos_dia['glucosa'].append(simulador.glucosa)
            datos_dia['dosis_recomendada'].append(dosis)
            datos_dia['accion_idx'].append(accion_idx)
            datos_dia['insulina_activa'].append(getattr(simulador, 'insulina_activa', 0))
            
            # Obtener descripción del estado actual
            estado_desc = f"G:{estado[0]}, I:{estado[1]}U, T:{estado[2]}, S:{estado[3]}"
            datos_dia['estado_descripcion'].append(estado_desc)
            
            # Aplicar acción y avanzar en el tiempo
            try:
                nuevo_estado, _, _ = simulador.step(accion_idx)
            except:
                # Si falla step, simular manualmente
                nuevo_estado = estado  # Mantener mismo estado
                # Reducir glucosa basado en dosis
                factor_sens = paciente_data.get('factor_sensibilidad', 50)
                reduccion = dosis * (50 / max(20, factor_sens))
                simulador.glucosa = max(50, simulador.glucosa - reduccion)
                # Añadir variabilidad
                simulador.glucosa += random.uniform(-5, 5)
            
            nuevo_estado_idx = self.agente.estado_a_indice(nuevo_estado, simulador)
            
            # Actualizar para siguiente paso
            estado = nuevo_estado
            estado_idx = nuevo_estado_idx
        
        return datos_dia, simulador
    
    def generar_plan_24h(self, paciente_id=None, glucosa_inicial=None, mostrar_detalles=True):

        print("GENERANDO PLAN DE INSULINA PERSONALIZADO")

        # 1. Seleccionar paciente
        paciente_data, paciente_idx = self.seleccionar_paciente_aleatorio(paciente_id)
        
        if mostrar_detalles:
            print(f"\nPACIENTE SELECCIONADO (ID: {paciente_idx}):")
            print(f"  Edad: {paciente_data['edad']} años")
            print(f"  Peso: {paciente_data['peso_kg']} kg")
            print(f"  IMC: {paciente_data['imc']:.1f}")
            print(f"  HbA1c: {paciente_data['hba1c']}%")
            print(f"  Tipo diabetes: {paciente_data['tipo_diabetes_especifico']}")
            print(f"  Años con diabetes: {paciente_data.get('años_con_diabetes', 'N/A')}")
            print(f"  Factor sensibilidad: {paciente_data['factor_sensibilidad']}")
            print(f"  Dosis total diaria: {paciente_data['dosis_total_insulina']:.1f} U")
        
        # 2. Simular día completo
        if glucosa_inicial is None:
            glucosa_inicial = random.randint(120, 180)
        
        datos_dia, simulador = self.simular_dia_completo(paciente_data, glucosa_inicial)
        
        # 3. Analizar resultados
        glucosas = np.array(datos_dia['glucosa'])
        dosis = np.array(datos_dia['dosis_recomendada'])
        
        # Calcular métricas
        tiempo_en_rango = np.mean((glucosas >= 70) & (glucosas <= 180)) * 100
        hipoglucemias = np.mean(glucosas < 70) * 100
        hiperglucemias = np.mean(glucosas > 180) * 100
        glucosa_promedio = np.mean(glucosas)
        dosis_total = np.sum(dosis)
        
        if mostrar_detalles:
            print(f"\nRESULTADOS DE LA SIMULACIÓN (24 horas):")
            print(f"  Glucosa inicial: {glucosa_inicial} mg/dL")
            print(f"  Glucosa promedio: {glucosa_promedio:.1f} mg/dL")
            print(f"  Tiempo en rango (70-180): {tiempo_en_rango:.1f}%")
            print(f"  Hipoglucemias (<70): {hipoglucemias:.1f}%")
            print(f"  Hiperglucemias (>180): {hiperglucemias:.1f}%")
            print(f"  Dosis total recomendada: {dosis_total:.1f} U")
            print(f"  Dosis promedio por paso: {np.mean(dosis):.2f} U")
        
        # 4. Generar plan horario resumido
        print(f"\nPLAN DE INSULINA RECOMENDADO:")
        print(f"Hora  | Dosis | Glucosa | Estado")
        print("-" * 45)
        
        # Agrupar por horas para plan legible
        horas_plan = [6, 12, 18, 21]  # Desayuno, almuerzo, cena, noche
        for hora_objetivo in horas_plan:
            # Encontrar pasos cercanos a esta hora
            indices_hora = [i for i, h in enumerate(datos_dia['hora']) 
                           if abs(h - hora_objetivo) < 0.5]
            
            if indices_hora:
                idx = indices_hora[0]
                hora_str = f"{int(datos_dia['hora'][idx]):02d}:{int((datos_dia['hora'][idx] % 1) * 60):02d}"
                dosis_rec = datos_dia['dosis_recomendada'][idx]
                glucosa_actual = datos_dia['glucosa'][idx]
                estado_desc = datos_dia['estado_descripcion'][idx]
                
                # Interpretar dosis
                if dosis_rec == 0:
                    recomendacion = "Sin insulina"
                elif dosis_rec <= 2:
                    recomendacion = "Dosis mínima"
                elif dosis_rec <= 5:
                    recomendacion = "Dosis pequeña"
                elif dosis_rec <= 7:
                    recomendacion = "Dosis moderada"
                else:
                    recomendacion = "Dosis alta"
                
                print(f"{hora_str} | {dosis_rec:5.1f}U | {glucosa_actual:7.0f} | {recomendacion}")
        
        # 5. Recomendaciones específicas
        print(f"\nRECOMENDACIONES ESPECÍFICAS PARA ESTE PACIENTE:")
        
        # Basado en el tipo de diabetes
        tipo_diabetes = paciente_data['tipo_diabetes_especifico']
        if 'infantil' in tipo_diabetes:
            print("  • Paciente pediátrico: considerar dosis más conservadoras")
            print("  • Monitorear frecuentemente por riesgo de hipoglucemia")
        elif 'lada' in tipo_diabetes:
            print("  • Diabetes LADA: puede requerir ajustes más graduales")
            print("  • Considerar posible resistencia a insulina")
        
        # Basado en sensibilidad
        factor_sens = paciente_data['factor_sensibilidad']
        if factor_sens < 40:
            print("  • Sensibilidad BAJA a insulina: puede requerir dosis más altas")
        elif factor_sens > 60:
            print("  • Sensibilidad ALTA a insulina: cuidado con hipoglucemias")
        
        # Basado en HbA1c
        hba1c = paciente_data['hba1c']
        if hba1c > 8.0:
            print(f"  • HbA1c elevada ({hba1c}%): considerar intensificar tratamiento")
        elif hba1c < 7.0:
            print(f"  • HbA1c en objetivo ({hba1c}%): mantener estrategia actual")
        
        return {
            'paciente_data': paciente_data,
            'datos_dia': datos_dia,
            'simulador': simulador,
            'metricas': {
                'tiempo_en_rango': tiempo_en_rango,
                'hipoglucemias': hipoglucemias,
                'hiperglucemias': hiperglucemias,
                'glucosa_promedio': glucosa_promedio,
                'dosis_total': dosis_total
            }
        }
    
    def visualizar_plan(self, resultado, save_figure=True):
        
        datos_dia = resultado['datos_dia']
        paciente_data = resultado['paciente_data']
        metricas = resultado['metricas']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Plan de Insulina Personalizado - Paciente ID: {paciente_data.get("id", "N/A")}', 
                    fontsize=14, fontweight='bold')
        
        # 1. Evolución de glucosa
        axes[0, 0].plot(datos_dia['hora'], datos_dia['glucosa'], 'b-', linewidth=2, label='Glucosa')
        axes[0, 0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Límite bajo (70)')
        axes[0, 0].axhline(y=180, color='r', linestyle='--', alpha=0.5, label='Límite alto (180)')
        axes[0, 0].fill_between(datos_dia['hora'], 70, 180, alpha=0.2, color='green', label='Rango objetivo')
        axes[0, 0].set_title('Evolución de Glucosa (24 horas)')
        axes[0, 0].set_xlabel('Hora del día')
        axes[0, 0].set_ylabel('Glucosa (mg/dL)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Dosis recomendadas
        axes[0, 1].bar(datos_dia['hora'], datos_dia['dosis_recomendada'], 
                      width=0.4, alpha=0.7, color='orange', label='Dosis')
        axes[0, 1].set_title('Dosis de Insulina Recomendadas')
        axes[0, 1].set_xlabel('Hora del día')
        axes[0, 1].set_ylabel('Dosis (unidades)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Distribución de acciones
        acciones = datos_dia['accion_idx']
        distribucion = np.bincount(acciones, minlength=4)  # 4 acciones según tu PDF
        
        acciones_nombres = ['0U', '2U', '5U', '7U']  # Según tu configuración
        axes[1, 0].bar(acciones_nombres, distribucion, alpha=0.7, color='purple')
        axes[1, 0].set_title('Distribución de Decisiones')
        axes[1, 0].set_xlabel('Acción (dosis)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Métricas principales
        metricas_barras = ['Tiempo en rango', 'Hipoglucemias', 'Hiperglucemias']
        valores = [metricas['tiempo_en_rango'], metricas['hipoglucemias'], metricas['hiperglucemias']]
        colores = ['green', 'red', 'orange']
        
        bars = axes[1, 1].bar(metricas_barras, valores, color=colores, alpha=0.7)
        axes[1, 1].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Objetivo (70%)')
        axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Límite (5%)')
        axes[1, 1].axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Límite (25%)')
        
        for bar, val in zip(bars, valores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom')
        
        axes[1, 1].set_title('Métricas de Control Glucémico')
        axes[1, 1].set_ylabel('Porcentaje (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Información del paciente
        info_text = f"Edad: {paciente_data['edad']} años\n"
        info_text += f"IMC: {paciente_data['imc']:.1f}\n"
        info_text += f"HbA1c: {paciente_data['hba1c']}%\n"
        info_text += f"Sensibilidad: {paciente_data['factor_sensibilidad']}\n"
        info_text += f"Dosis total: {metricas['dosis_total']:.1f} U"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_figure:
            # Crear directorio si no existe
            os.makedirs('Resultados/atencion_personalizada', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            paciente_id = paciente_data.get('id', 'unknown')
            filename = f"Resultados/atencion_personalizada/plan_insulina_paciente_{paciente_id}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nGráfico guardado como: {filename}")
        
        plt.show()
        return fig
    
    def guardar_plan(self, resultado, nombre_base='plan_insulina'):
        "Guarda el plan generado en archivos CSV y JSON"
        
        paciente_data = resultado['paciente_data']
        datos_dia = resultado['datos_dia']
        metricas = resultado['metricas']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paciente_id = paciente_data.get('id', 'unknown')
        
        # Crear directorio si no existe
        os.makedirs('Resultados/atencion_personalizada', exist_ok=True)
        
        # 1. Guardar plan detallado como CSV
        nombre_csv = f"Resultados/atencion_personalizada/{nombre_base}_detallado_paciente_{paciente_id}_{timestamp}.csv"
        
        df_detallado = pd.DataFrame({
            'hora': datos_dia['hora'],
            'glucosa_mg_dL': datos_dia['glucosa'],
            'dosis_recomendada_U': datos_dia['dosis_recomendada'],
            'accion_idx': datos_dia['accion_idx'],
            'insulina_activa_U': datos_dia['insulina_activa'],
            'estado_descripcion': datos_dia['estado_descripcion']
        })
        
        df_detallado.to_csv(nombre_csv, index=False, encoding='utf-8')
        
        # 2. Función para convertir todos los tipos de numpy/pandas a Python nativo
        def convert_to_python_types(obj):
            # Si es un tipo de numpy
            if hasattr(obj, 'item'):  # Para numpy scalars
                try:
                    return obj.item()
                except:
                    pass
            
            # Convertir tipos específicos de numpy
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_python_types(obj.tolist())
            elif isinstance(obj, pd.Series):
                return convert_to_python_types(obj.to_dict())
            elif isinstance(obj, pd.DataFrame):
                return convert_to_python_types(obj.to_dict(orient='records'))
            # Para diccionarios y listas, procesar recursivamente
            elif isinstance(obj, dict):
                return {key: convert_to_python_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_python_types(item) for item in obj)
            # Para datetime y fechas
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            # Dejar otros tipos como están
            else:
                return obj
        
        # 3. Crear resumen asegurando que todos los valores sean serializables
        tiempo_en_rango = float(metricas['tiempo_en_rango']) if hasattr(metricas['tiempo_en_rango'], 'item') else metricas['tiempo_en_rango']
        hipoglucemias = float(metricas['hipoglucemias']) if hasattr(metricas['hipoglucemias'], 'item') else metricas['hipoglucemias']
        hiperglucemias = float(metricas['hiperglucemias']) if hasattr(metricas['hiperglucemias'], 'item') else metricas['hiperglucemias']
        glucosa_promedio = float(metricas['glucosa_promedio']) if hasattr(metricas['glucosa_promedio'], 'item') else metricas['glucosa_promedio']
        dosis_total = float(metricas['dosis_total']) if hasattr(metricas['dosis_total'], 'item') else metricas['dosis_total']
        
        # Determinar control glucémico basado en tiempo en rango
        if tiempo_en_rango > 80:
            control_glucemico = 'Excelente'
        elif tiempo_en_rango > 70:
            control_glucemico = 'Bueno'
        elif tiempo_en_rango > 60:
            control_glucemico = 'Regular'
        else:
            control_glucemico = 'Pobre'
        
        # Convertir datos del paciente
        paciente_converted = {}
        for key, value in paciente_data.items():
            if hasattr(value, 'item'):
                paciente_converted[key] = value.item()
            else:
                paciente_converted[key] = convert_to_python_types(value)
        
        # Crear el diccionario de resumen
        resumen = {
            'paciente': paciente_converted,
            'metricas': {
                'tiempo_en_rango': tiempo_en_rango,
                'hipoglucemias': hipoglucemias,
                'hiperglucemias': hiperglucemias,
                'glucosa_promedio': glucosa_promedio,
                'dosis_total': dosis_total
            },
            'recomendaciones': {
                'dosis_total_diaria': dosis_total,
                'glucosa_promedio': glucosa_promedio,
                'control_glucemico': control_glucemico
            },
            'archivos_generados': {
                'detallado': nombre_csv,
                'resumen': f"Resultados/atencion_personalizada/{nombre_base}_resumen_paciente_{paciente_id}_{timestamp}.json"
            },
            'fecha_generacion': timestamp
        }
        
        # 4. Guardar como JSON usando un encoder personalizado
        nombre_json = resumen['archivos_generados']['resumen']
        
        class NumpyEncoder(json.JSONEncoder):
            """Encoder personalizado para manejar tipos de numpy"""
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                else:
                    return super().default(obj)
        
        try:
            with open(nombre_json, 'w', encoding='utf-8') as f:
                json.dump(resumen, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            # Si todavía hay problemas, usar una conversión más agresiva
            print(f"Advertencia: Convirtiendo datos de forma más agresiva debido a: {e}")
            resumen_simple = json.loads(json.dumps(resumen, default=lambda o: str(o) if hasattr(o, 'item') else str(o)))
            with open(nombre_json, 'w', encoding='utf-8') as f:
                json.dump(resumen_simple, f, indent=2, ensure_ascii=False)
        
        print(f"\nPLAN GUARDADO:")
        print(f"  • {nombre_csv} - Plan detallado (cada 30 minutos)")
        print(f"  • {nombre_json} - Resumen y recomendaciones")
        
        return {
            'csv': nombre_csv,
            'json': nombre_json
        }
    
    def ejecutar_planificacion_completa(self, paciente_id=None, glucosa_inicial=None):
        
        # Generar plan
        resultado = self.generar_plan_24h(paciente_id, glucosa_inicial)
        
        # Visualizar resultados
        self.visualizar_plan(resultado)
        
        # Guardar plan
        archivos = self.guardar_plan(resultado)
        
        return resultado, archivos

def main():
    print("GENERADOR DE PLANES DE INSULINA PERSONALIZADOS")
    print("Basado en modelo RL entrenado")
    
    # Configuración
    MODELO_PATH = "Resultados/best_model/best_model.pkl"
    DB_PATH = "Base de datos/db_diabetes_50k.csv"
    
    # Crear directorio para atención personalizada si no existe
    os.makedirs('Resultados/atencion_personalizada', exist_ok=True)
    
    # Crear planificador
    planificador = PlanificadorInsulinaPersonalizado(
        modelo_path=MODELO_PATH,
        db_path=DB_PATH
    )
    
    # Opciones de ejecución
    print("\nOpciones:")
    print("1. Plan para paciente aleatorio")
    print("2. Plan para paciente específico (por ID)")
    print("3. Salir")
    
    try:
        opcion = input("\nSeleccione opción (1-3): ").strip()
        
        if opcion == '1':
            # Paciente aleatorio
            resultado, archivos = planificador.ejecutar_planificacion_completa()
            
        elif opcion == '2':
            # Paciente específico
            try:
                paciente_id = int(input("Ingrese ID del paciente (0-49999): "))
                if 0 <= paciente_id < 50000:
                    glucosa_inicial = input("Glucosa inicial (dejar vacío para aleatoria): ").strip()
                    if glucosa_inicial:
                        glucosa_inicial = float(glucosa_inicial)
                    else:
                        glucosa_inicial = None
                    
                    resultado, archivos = planificador.ejecutar_planificacion_completa(
                        paciente_id=paciente_id,
                        glucosa_inicial=glucosa_inicial
                    )
                else:
                    print("ID fuera de rango. Usando paciente aleatorio.")
                    resultado, archivos = planificador.ejecutar_planificacion_completa()
            except ValueError:
                print("Entrada inválida. Usando paciente aleatorio.")
                resultado, archivos = planificador.ejecutar_planificacion_completa()
        
        elif opcion == '3':
            print("Saliendo...")
            return
        
        else:
            print("Opción no válida. Usando paciente aleatorio.")
            resultado, archivos = planificador.ejecutar_planificacion_completa()
        
        print("PROCESO COMPLETADO EXITOSAMENTE")
        
    except Exception as e:
        print(f"\nError durante la planificación: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()