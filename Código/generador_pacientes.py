import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

def generar_dataset_antropometrico_coherente(nombre_archivo="db_try_50k.csv"):
    
    # Tamaño del dataset: 50,000 pacientes, todos con diabetes tipo 1
    TOTAL_PACIENTES = 50000
    
    # Distribución por género en diabetes tipo 1
    DIABETES_HOMBRES = 25000
    DIABETES_MUJERES = 25000
    
    # Límite de desviaciones estándar para valores antropométricos: ±2 desviaciones (95% de la población)
    # Esto evita valores extremos no realistas
    DESVIACIONES_PERMITIDAS = 2
    
    # Tablas de crecimiento antropométrico basadas en percentiles WHO/CDC
    # Formato por edad: (altura_promedio_cm, desviacion_altura_cm, peso_promedio_kg, desviacion_peso_kg)
    # Estos valores representan la media y desviación estándar para cada edad y género
    TABLAS_CRECIMIENTO = {
        'hombre': {
            0: (50.5, 2.5, 3.5, 0.5),
            1: (76.0, 3.0, 9.5, 1.2),
            2: (87.0, 3.5, 12.5, 1.5),
            3: (95.0, 4.0, 14.5, 1.8),
            4: (102.0, 4.5, 16.5, 2.0),
            5: (109.0, 5.0, 18.5, 2.5),
            6: (116.0, 5.5, 21.0, 3.0),
            7: (122.0, 6.0, 23.5, 3.5),
            8: (128.0, 6.5, 26.5, 4.0),
            9: (133.0, 7.0, 29.5, 4.5),
            10: (138.0, 7.5, 32.5, 5.0),
            11: (143.0, 8.0, 36.0, 6.0),
            12: (149.0, 8.5, 40.0, 7.0),
            13: (156.0, 9.0, 45.0, 8.0),
            14: (163.0, 9.0, 51.0, 9.0),
            15: (169.0, 9.0, 57.0, 10.0),
            16: (173.0, 8.5, 62.0, 10.0),
            17: (175.0, 8.0, 66.0, 10.0),
            18: (176.0, 7.5, 68.0, 10.0),
            20: (177.0, 7.0, 73.0, 11.0),
            25: (177.0, 7.0, 78.0, 12.0),
            30: (177.0, 7.0, 82.0, 13.0),
            40: (176.0, 7.0, 85.0, 14.0),
            50: (175.0, 7.0, 86.0, 14.0),
            60: (174.0, 7.0, 85.0, 13.0),
            70: (172.0, 7.0, 82.0, 12.0),
            80: (170.0, 7.0, 79.0, 11.0),
            85: (169.0, 7.0, 77.0, 10.0)
        },
        'mujer': {
            0: (49.5, 2.5, 3.4, 0.5),
            1: (74.0, 3.0, 9.0, 1.1),
            2: (86.0, 3.5, 12.0, 1.4),
            3: (94.0, 4.0, 14.0, 1.7),
            4: (101.0, 4.5, 16.0, 1.9),
            5: (108.0, 5.0, 18.0, 2.3),
            6: (115.0, 5.5, 20.5, 2.8),
            7: (121.0, 6.0, 23.0, 3.3),
            8: (127.0, 6.5, 26.0, 4.0),
            9: (132.0, 7.0, 29.0, 4.5),
            10: (138.0, 7.5, 32.5, 5.0),
            11: (144.0, 8.0, 36.5, 6.0),
            12: (150.0, 8.5, 41.0, 7.0),
            13: (156.0, 8.5, 46.0, 8.0),
            14: (160.0, 8.0, 50.0, 8.0),
            15: (162.0, 7.5, 53.0, 8.0),
            16: (163.0, 7.0, 55.0, 8.0),
            17: (164.0, 7.0, 56.0, 8.0),
            18: (164.0, 7.0, 57.0, 8.0),
            20: (164.0, 7.0, 60.0, 9.0),
            25: (164.0, 7.0, 63.0, 10.0),
            30: (164.0, 7.0, 66.0, 11.0),
            40: (163.0, 7.0, 68.0, 12.0),
            50: (162.0, 7.0, 69.0, 12.0),
            60: (161.0, 7.0, 68.0, 11.0),
            70: (159.0, 7.0, 66.0, 10.0),
            80: (157.0, 7.0, 64.0, 9.0),
            85: (156.0, 7.0, 62.0, 9.0)
        }
    }
    
    def obtener_valores_antropometricos(edad, genero):
        """Obtiene los valores antropométricos (altura y peso) para una edad y género específicos
        
        Args:
            edad: Edad en años (entre 0 y 85)
            genero: 'hombre' o 'mujer'
            
        Returns:
            Tupla con: (altura_media, desviacion_altura, peso_medio, desviacion_peso)
            Si la edad no está exactamente en la tabla, interpola linealmente entre las edades más cercanas
        """
        edad = min(max(edad, 0), 85)
        
        edades_disponibles = sorted(TABLAS_CRECIMIENTO[genero].keys())
        
        # Si la edad está exactamente en la tabla, devuelve esos valores
        if edad in edades_disponibles:
            return TABLAS_CRECIMIENTO[genero][edad]
        
        # Encuentra las edades más cercanas para interpolación
        edad_inferior = max([e for e in edades_disponibles if e <= edad])
        edad_superior = min([e for e in edades_disponibles if e >= edad])
        
        # Obtiene los valores para las edades inferior y superior
        altura_inf, altura_sd_inf, peso_inf, peso_sd_inf = TABLAS_CRECIMIENTO[genero][edad_inferior]
        altura_sup, altura_sd_sup, peso_sup, peso_sd_sup = TABLAS_CRECIMIENTO[genero][edad_superior]
        
        # Calcula factor de interpolación lineal
        if edad_superior > edad_inferior:
            factor = (edad - edad_inferior) / (edad_superior - edad_inferior)
        else:
            factor = 0
        
        # Interpola linealmente todos los valores
        altura_interpolada = altura_inf + (altura_sup - altura_inf) * factor
        altura_sd_interpolada = altura_sd_inf + (altura_sd_sup - altura_sd_inf) * factor
        peso_interpolado = peso_inf + (peso_sup - peso_inf) * factor
        peso_sd_interpolado = peso_sd_inf + (peso_sd_sup - peso_sd_inf) * factor
        
        return altura_interpolada, altura_sd_interpolada, peso_interpolado, peso_sd_interpolado
    
    def generar_peso_altura_coherente(edad, genero, imc_objetivo=None):
        """Genera peso y altura coherentes con la edad, género y posible IMC objetivo
        
        Args:
            edad: Edad en años
            genero: 'hombre' o 'mujer'
            imc_objetivo: IMC deseado (opcional). Si se especifica, ajusta el peso para alcanzarlo
            
        Returns:
            Tupla con: (peso_kg, altura_cm, imc_real) todos redondeados a 1 decimal
        """
        # Obtiene los valores antropométricos base para esta edad y género
        altura_media, desviacion_altura, peso_medio, desviacion_peso = obtener_valores_antropometricos(edad, genero)
        
        # Genera altura con distribución normal alrededor de la media
        altura_generada = np.random.normal(altura_media, desviacion_altura)
        
        if imc_objetivo is not None:
            # Si hay IMC objetivo, calcula el peso necesario para alcanzarlo con la altura generada
            peso_calculado = imc_objetivo * ((altura_generada/100) ** 2)
        else:
            # Genera peso base con distribución normal
            peso_base = np.random.normal(peso_medio, desviacion_peso)
            
            # Ajusta el peso según la altura (correlación positiva altura-peso)
            correlacion_altura_peso = 0.6  # Personas más altas tienden a pesar más
            ajuste_por_altura = (altura_generada - altura_media) / desviacion_altura * desviacion_peso * correlacion_altura_peso
            peso_calculado = peso_base + ajuste_por_altura
        
        # Limita los valores dentro de ±DESVIACIONES_PERMITIDAS desviaciones estándar
        # Esto evita valores extremos no realistas
        altura_generada = max(altura_media - DESVIACIONES_PERMITIDAS*desviacion_altura, 
                            min(altura_generada, altura_media + DESVIACIONES_PERMITIDAS*desviacion_altura))
        peso_calculado = max(peso_medio - DESVIACIONES_PERMITIDAS*desviacion_peso, 
                           min(peso_calculado, peso_medio + DESVIACIONES_PERMITIDAS*desviacion_peso))
        
        # Límites absolutos razonables para humanos
        altura_generada = max(45, min(210, altura_generada))  # Entre 45cm (recién nacido) y 210cm
        peso_calculado = max(2.5, min(180, peso_calculado))   # Entre 2.5kg y 180kg
        
        # Calcula el IMC real
        imc_calculado = peso_calculado / ((altura_generada/100) ** 2)
        
        # Retorna valores redondeados a 1 decimal
        return round(peso_calculado, 1), round(altura_generada, 1), round(imc_calculado, 1)
    
    def determinar_perfil_metabolico(edad, genero):
        """Determina rangos metabólicos coherentes según diabetes, edad, género
        
        Args:
            edad: Edad en años
            genero: 'hombre' o 'mujer'
            
        Returns:
            Diccionario con: imc_objetivo, rango_hba1c, rango_glucosa_ayunas, categoria_control
            Los rangos están basados en guías clínicas ADA (American Diabetes Association)
        """
        # Diabetes tipo 1 - rangos según calidad de control
        # Distribución en campana: mayoría con control regular-bueno
        
        # AJUSTE IMPORTANTE: Para niños con diabetes, rangos diferentes
        if edad < 18:
            # Niños: objetivos pediátricos diferentes
            niveles_control = random.choices(
                ['excelente', 'bueno', 'regular', 'malo'],
                weights=[0.10, 0.40, 0.40, 0.10]  # Más niños en control regular-bueno
            )[0]
            
            if niveles_control == 'excelente':
                rango_imc = (16.0, 22.0)  # IMC pediátrico normal más bajo
                rango_hba1c = (6.5, 7.5)   # Objetivo pediátrico ADA <7.5%
                rango_glucosa_ayunas = (80, 150)  # Mayor rango aceptable en niños
                
            elif niveles_control == 'bueno':
                rango_imc = (17.0, 23.0)
                rango_hba1c = (7.5, 8.5)
                rango_glucosa_ayunas = (120, 200)
                
            elif niveles_control == 'regular':
                rango_imc = (18.0, 25.0)
                rango_hba1c = (8.5, 10.0)
                rango_glucosa_ayunas = (150, 250)
                
            else:  # malo
                rango_imc = (19.0, 27.0)
                rango_hba1c = (10.0, 13.0)  # Niños con control muy malo
                rango_glucosa_ayunas = (200, 300)
                
        else:
            # Adultos: rangos estándar
            niveles_control = random.choices(
                ['excelente', 'bueno', 'regular', 'malo'],
                weights=[0.15, 0.35, 0.35, 0.15]
            )[0]
            
            if niveles_control == 'excelente':
                rango_imc = (20.0, 25.0)  # IMC normal
                rango_hba1c = (6.0, 7.0)   # Objetivo ADA <7%
                rango_glucosa_ayunas = (80, 130)  # Objetivo ADA: 80-130 mg/dL
                
            elif niveles_control == 'bueno':
                rango_imc = (21.0, 27.0)  # Normopeso a sobrepeso leve
                rango_hba1c = (7.0, 7.8)
                rango_glucosa_ayunas = (100, 160)
                
            elif niveles_control == 'regular':
                rango_imc = (24.0, 29.0)  # Sobrepeso
                rango_hba1c = (7.8, 8.5)
                rango_glucosa_ayunas = (140, 200)
                
            else:  # malo
                rango_imc = (27.0, 35.0)  # Obesidad
                rango_hba1c = (8.5, 12.0)
                rango_glucosa_ayunas = (180, 300)
        
        # El IMC tiende a aumentar con la edad (0.3-0.5 kg/m² por década)
        imc_base = random.uniform(*rango_imc)
        if edad > 50:
            imc_base += random.uniform(0, 2)  # +0-2 en mayores
        elif edad > 30:
            imc_base += random.uniform(0, 1)  # +0-1 en adultos
        
        imc_objetivo = min(imc_base, 45)
        
        return {
            'imc_objetivo': imc_objetivo,
            'rango_hba1c': rango_hba1c,
            'rango_glucosa_ayunas': rango_glucosa_ayunas,
            'categoria_control': niveles_control
        }
    
    def asignar_tipo_ejercicio_coherente(nivel_actividad, imc, edad):
        """Asigna tipo de ejercicio coherente con nivel de actividad, IMC y edad
        
        Args:
            nivel_actividad: 1-5 (1=muy baja, 5=muy alta)
            imc: Índice de masa corporal
            edad: Edad en años
            
        Returns:
            0: Sedentario (sin ejercicio estructurado)
            1: Baja intensidad (caminar, yoga suave)
            2: Moderada intensidad (ciclismo moderado, natación)
            3: Alta intensidad (running, HIIT, deportes competitivos)
            
        La asignación sigue patrones reales: personas con mayor IMC y edad
        tienden a hacer ejercicio menos intenso
        """
        if nivel_actividad <= 2 or imc >= 30:
            # Sedentarios o con obesidad: tipo 0 o 1
            if random.random() < 0.7:
                return 0  # Sedentario (sin ejercicio)
            else:
                return 1  # Baja intensidad
        
        elif nivel_actividad == 3:
            # Actividad moderada: tipo 1 o 2
            if imc < 25:
                # Normopeso: más probabilidad de ejercicio moderado
                return random.choices([1, 2], weights=[0.4, 0.6])[0]
            else:
                # Sobrepeso: más probabilidad de ejercicio suave
                return random.choices([1, 2], weights=[0.7, 0.3])[0]
        
        elif nivel_actividad == 4:
            # Actividad alta: tipo 2 o 3
            if edad < 40 and imc < 25:
                # Jóvenes normopesos: pueden hacer ejercicio intenso
                return random.choices([2, 3], weights=[0.5, 0.5])[0]
            else:
                # Mayores o con sobrepeso: más moderado
                return random.choices([2, 3], weights=[0.7, 0.3])[0]
        
        else:  # nivel_actividad == 5
            # Actividad muy alta: tipo 3 predominante
            if edad < 50:
                return random.choices([2, 3], weights=[0.3, 0.7])[0]
            else:
                return random.choices([2, 3], weights=[0.6, 0.4])[0]
    
    def determinar_tipo_diabetes_especifico(edad_diagnostico):
        """        
        Args:
            edad_diagnostico: Edad en años cuando se diagnosticó la diabetes
            
        Returns:
            String con la clasificación específica:
            - 'diabetes_inicio_infantil': diagnóstico antes de los 6 años
            - 'diabetes_tipo1_clasica': diagnóstico entre 6-19 años
            - 'diabetes_diagnostico_tardio': diagnóstico entre 20-34 años
            - 'diabetes_lada': diagnóstico después de los 35 años
        """
        if edad_diagnostico < 6:
            return 'diabetes_inicio_infantil'
        elif edad_diagnostico < 20:
            return 'diabetes_tipo1_clasica'
        elif edad_diagnostico < 35:
            return 'diabetes_diagnostico_tardio'
        else:
            return 'diabetes_lada'
    
    def generar_paciente_diabetico(id_counter, genero):
        """
        
        Args:
            id_counter: Identificador único del paciente
            genero: 'hombre' o 'mujer'
            
        Returns:
            Lista con todos los datos del paciente en el orden definido
        """
        # 1. GENERACIÓN DE EDAD - Distribuciones para diabéticos
        # Diabetes tipo 1: distribución bimodal
        # 70% diagnóstico infantil/adolescente, 30% diagnóstico adulto (LADA)
        if random.random() < 0.7:
            # Diagnóstico infantil/adolescente (pico entre 5-15 años)
            if genero == 'hombre':
                edad_generada = np.random.normal(12, 8)  # Media 12, desviación 8
            else:
                edad_generada = np.random.normal(11, 7)
        else:
            # Diagnóstico adulto/LADA (pico entre 30-40 años)
            if genero == 'hombre':
                edad_generada = np.random.normal(35, 8)
            else:
                edad_generada = np.random.normal(33, 8)
        
        # Limita edad entre 1 y 85 años
        edad_generada = max(1, min(85, int(edad_generada)))
        
        # 2. DETERMINACIÓN DEL PERFIL METABÓLICO
        # Define rangos de HbA1c, glucosa e IMC objetivo según edad y género
        perfil = determinar_perfil_metabolico(edad_generada, genero)
        
        # 3. GENERACIÓN DE ANTROPOMETRÍA (peso, altura, IMC)
        # Peso y altura coherentes con edad, género e IMC objetivo
        peso_kg, altura_cm, imc_real = generar_peso_altura_coherente(
            edad_generada, genero, perfil['imc_objetivo']
        )
        
        # 4. GENERACIÓN DE VARIABLES DE ACTIVIDAD FÍSICA - VERSIÓN MEJORADA
        if edad_generada < 5:
            # Bebés y niños pequeños (0-4 años): actividad física básica, no ejercicio estructurado
            nivel_actividad = random.choices([1, 2], weights=[0.8, 0.2])[0]
            horas_ejercicio_semana = random.uniform(0, 2)
            tipo_ejercicio_predominante = 0
        
        elif edad_generada < 18:
            # Niños y adolescentes (5-17 años): actividad variable según IMC
            if imc_real < 25:
                nivel_actividad = random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]
                horas_ejercicio_semana = random.uniform(2, 8)
            elif imc_real < 30:
                nivel_actividad = random.choices([2, 3], weights=[0.6, 0.4])[0]
                horas_ejercicio_semana = random.uniform(1, 5)
            else:
                nivel_actividad = random.choices([1, 2], weights=[0.7, 0.3])[0]
                horas_ejercicio_semana = random.uniform(0, 3)
            
            if nivel_actividad >= 3:
                tipo_ejercicio_predominante = random.choices([1, 2], weights=[0.5, 0.5])[0]
            else:
                tipo_ejercicio_predominante = random.choices([0, 1], weights=[0.6, 0.4])[0]
        
        elif edad_generada < 65:
            # Adultos (18-64 años): modelo con coherencia IMC-actividad
            if imc_real < 25:
                nivel_actividad = random.choices([3, 4, 5], weights=[0.3, 0.5, 0.2])[0]
                horas_ejercicio_semana = random.uniform(3, 10)
            elif imc_real < 30:
                nivel_actividad = random.choices([2, 3, 4], weights=[0.4, 0.5, 0.1])[0]
                horas_ejercicio_semana = random.uniform(2, 6)
            else:
                nivel_actividad = random.choices([1, 2, 3], weights=[0.5, 0.4, 0.1])[0]
                horas_ejercicio_semana = random.uniform(0, 3)
            
            tipo_ejercicio_predominante = asignar_tipo_ejercicio_coherente(
                nivel_actividad, imc_real, edad_generada
            )
        
        else:
            # Adultos mayores (65+ años): actividad más conservadora
            if imc_real < 25:
                nivel_actividad = random.choices([2, 3, 4], weights=[0.5, 0.4, 0.1])[0]
                horas_ejercicio_semana = random.uniform(1, 5)
            elif imc_real < 30:
                nivel_actividad = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                horas_ejercicio_semana = random.uniform(0, 4)
            else:
                nivel_actividad = random.choices([1, 2], weights=[0.8, 0.2])[0]
                horas_ejercicio_semana = random.uniform(0, 2)
            
            if nivel_actividad >= 3:
                tipo_ejercicio_predominante = random.choices([1, 2], weights=[0.7, 0.3])[0]
            else:
                tipo_ejercicio_predominante = random.choices([0, 1], weights=[0.6, 0.4])[0]
        
        # 5. GENERACIÓN DE VARIABLES METABÓLICAS
        # HbA1c generado dentro del rango definido por el perfil
        hba1c_generado = np.random.normal(
            np.mean(perfil['rango_hba1c']), 
            (perfil['rango_hba1c'][1] - perfil['rango_hba1c'][0]) / 4
        )
        hba1c_generado = max(perfil['rango_hba1c'][0], 
                           min(perfil['rango_hba1c'][1], hba1c_generado))
        
        # Glucosa en ayunas coherente con HbA1c
        glucosa_ayunas_generada = np.random.normal(
            np.mean(perfil['rango_glucosa_ayunas']),
            (perfil['rango_glucosa_ayunas'][1] - perfil['rango_glucosa_ayunas'][0]) / 4
        )
        glucosa_ayunas_generada = max(perfil['rango_glucosa_ayunas'][0], 
                                    min(perfil['rango_glucosa_ayunas'][1], glucosa_ayunas_generada))
        
        # Glucosa postprandial: mayor aumento en diabéticos, ajustado por calidad de control
        if perfil['categoria_control'] == 'excelente':
            factor_aumento_post = random.uniform(1.2, 1.6)  # Buen control
        elif perfil['categoria_control'] == 'bueno':
            factor_aumento_post = random.uniform(1.4, 1.8)
        elif perfil['categoria_control'] == 'regular':
            factor_aumento_post = random.uniform(1.6, 2.0)
        else:  # malo
            factor_aumento_post = random.uniform(1.8, 2.5)
        
        glucosa_post_generada = glucosa_ayunas_generada * factor_aumento_post
        # Límites más estrictos para evitar valores extremos
        if edad_generada < 18:
            limite_max = 350  # Niños diabéticos: límite más bajo
        else:
            limite_max = 400  # Adultos diabéticos
        
        glucosa_post_generada = max(glucosa_ayunas_generada + 10, min(limite_max, glucosa_post_generada))
        
        # 6. CÁLCULO DE AÑOS CON DIABETES - AJUSTADO
        # Edad de diagnóstico coherente con edad actual y realista
        if edad_generada < 20:
            # Niños: diagnóstico más realista (raro <1 año)
            if edad_generada < 2:
                # Diabetes neonatal muy rara - solo 5% de casos infantiles
                if random.random() < 0.05:
                    edad_diagnostico = random.uniform(0.3, 1.0)
                else:
                    edad_diagnostico = random.uniform(1.5, min(15, edad_generada - 0.5))
            else:
                edad_diagnostico = np.random.normal(7, 4)
            edad_diagnostico = max(0.5, min(edad_generada - 0.5, edad_diagnostico))
        elif edad_generada < 40:
            if random.random() < 0.7:
                edad_diagnostico = np.random.normal(12, 5)
            else:
                edad_diagnostico = np.random.normal(25, 8)
            edad_diagnostico = max(1, min(edad_generada - 1, edad_diagnostico))
        else:
            edad_diagnostico = np.random.normal(35, 10)
            edad_diagnostico = max(1, min(edad_generada - 1, edad_diagnostico))
        
        años_con_diabetes = max(0, edad_generada - edad_diagnostico)
        
        # 7. RACIONES DE CARBOHIDRATOS - basadas en requerimientos calóricos
        if genero == 'hombre':
            raciones_base = 80
        else:
            raciones_base = 65
        
        if nivel_actividad >= 4:
            raciones_base *= 1.2
        elif nivel_actividad <= 2:
            raciones_base *= 0.9
        
        if edad_generada < 18:
            raciones_base *= 0.8
        elif edad_generada > 60:
            raciones_base *= 0.9
        
        raciones_carbohidratos = np.random.normal(raciones_base, raciones_base * 0.2)
        raciones_carbohidratos = max(30, min(150, raciones_carbohidratos))
        
        # 8. TRATAMIENTO CON INSULINA - AJUSTADO PARA COHERENCIA
        # Dosis basada en peso, edad y calidad de control - VALORES MÁS REALISTAS
        if edad_generada < 18:
            # Niños: requerimientos diferentes
            if perfil['categoria_control'] == 'excelente':
                factor_insulina_por_peso = random.uniform(0.6, 0.9)  # 0.6-0.9 U/kg
            elif perfil['categoria_control'] == 'bueno':
                factor_insulina_por_peso = random.uniform(0.7, 1.0)  # 0.7-1.0 U/kg
            elif perfil['categoria_control'] == 'regular':
                factor_insulina_por_peso = random.uniform(0.8, 1.2)  # 0.8-1.2 U/kg
            else:  # malo
                factor_insulina_por_peso = random.uniform(0.9, 1.4)  # 0.9-1.4 U/kg
        else:
            # Adultos
            if perfil['categoria_control'] == 'excelente':
                factor_insulina_por_peso = random.uniform(0.5, 0.7)  # 0.5-0.7 U/kg
            elif perfil['categoria_control'] == 'bueno':
                factor_insulina_por_peso = random.uniform(0.6, 0.8)  # 0.6-0.8 U/kg
            elif perfil['categoria_control'] == 'regular':
                factor_insulina_por_peso = random.uniform(0.7, 0.9)  # 0.7-0.9 U/kg
            else:  # malo
                factor_insulina_por_peso = random.uniform(0.8, 1.1)  # 0.8-1.1 U/kg
        
        dosis_total_insulina = peso_kg * factor_insulina_por_peso
        
        # Distribución basal/bolo: 40-60% basal
        proporcion_basal = random.uniform(0.40, 0.60)
        dosis_basal = dosis_total_insulina * proporcion_basal
        dosis_bolo = dosis_total_insulina - dosis_basal
        
        # Ratio insulina/carbohidratos: 500/dosis_total (regla del 500)
        ratio_ic = 500 / dosis_total_insulina if dosis_total_insulina > 0 else 0
        
        # Factor de sensibilidad: 1800/dosis_total (regla del 1800)
        factor_sensibilidad = 1800 / dosis_total_insulina if dosis_total_insulina > 0 else 0
        
        # 9. CLASIFICACIÓN DEL PACIENTE
        # Determina el tipo específico de diabetes
        tipo_diabetes_especifico = determinar_tipo_diabetes_especifico(edad_diagnostico)
        
        # Retorna todos los datos en el orden definido
        return [
            # 1. IDENTIFICACIÓN Y CARACTERÍSTICAS BÁSICAS
            id_counter, genero, edad_generada,
            
            # 2. ANTROPOMETRÍA Y COMPOSICIÓN CORPORAL
            peso_kg, altura_cm, imc_real,
            
            # 3. ACTIVIDAD FÍSICA
            nivel_actividad, round(horas_ejercicio_semana, 1), tipo_ejercicio_predominante,
            
            # 4. METABOLISMO GLUCÉMICO Y TRATAMIENTO
            round(años_con_diabetes, 1),
            round(hba1c_generado, 1),
            round(glucosa_ayunas_generada), round(glucosa_post_generada),
            round(glucosa_post_generada - glucosa_ayunas_generada, 1),
            round(raciones_carbohidratos),
            round(dosis_total_insulina, 1),
            round(dosis_basal, 1),
            round(dosis_bolo, 1),
            round(ratio_ic, 1),
            round(factor_sensibilidad, 1),
            
            # 5. CLASIFICACIÓN
            tipo_diabetes_especifico
        ]

    # GENERACIÓN DEL DATASET COMPLETO
    
    datos_totales = []
    contador_id = 1
    
    print(f"Generando dataset de 50,000 registros (todos con diabetes tipo 1)...")
    
    # 1. HOMBRES CON DIABETES TIPO 1 (22,000 pacientes)
    print(f"Generando {DIABETES_HOMBRES} hombres con diabetes...")
    for i in range(DIABETES_HOMBRES):
        if i % 2000 == 0:
            print(f"  Progreso: {i}/{DIABETES_HOMBRES}")
        datos_totales.append(generar_paciente_diabetico(contador_id, 'hombre'))
        contador_id += 1
    
    # 2. MUJERES CON DIABETES TIPO 1 (28,000 pacientes)
    print(f"Generando {DIABETES_MUJERES} mujeres con diabetes...")
    for i in range(DIABETES_MUJERES):
        if i % 2000 == 0:
            print(f"  Progreso: {i}/{DIABETES_MUJERES}")
        datos_totales.append(generar_paciente_diabetico(contador_id, 'mujer'))
        contador_id += 1
    

    # CREACIÓN DEL DATAFRAME CON NOMBRES DE COLUMNAS

    
    nombres_columnas = [
        # 1. IDENTIFICACIÓN Y CARACTERÍSTICAS BÁSICAS
        'id',                # Identificador único del paciente
        'genero',            # 'hombre' o 'mujer'
        'edad',              # Edad en años (entero)
        
        # 2. ANTROPOMETRÍA Y COMPOSICIÓN CORPORAL
        'peso_kg',           # Peso en kilogramos (1 decimal)
        'altura_cm',         # Altura en centímetros (1 decimal)
        'imc',               # Índice de masa corporal (1 decimal)
        
        # 3. ACTIVIDAD FÍSICA
        'nivel_actividad_fisica',  # Escala 1-5 (1=muy baja, 5=muy alta)
        'horas_ejercicio_semana',  # Horas de ejercicio por semana (1 decimal)
        'tipo_ejercicio',          # 0=sedentario, 1=baja, 2=moderada, 3=alta intensidad
        
        # 4. METABOLISMO GLUCÉMICO Y TRATAMIENTO
        'años_con_diabetes',      # Años desde diagnóstico (1 decimal)
        'hba1c',                  # Hemoglobina glicosilada en % (1 decimal)
        'glucosa_en_ayunas',      # Glucosa sanguínea en ayunas mg/dL (entero)
        'glucosa_posprandial',    # Glucosa 2 horas post-comida mg/dL (entero)
        'diferencial_glucosa',    # Diferencia entre postprandial y ayunas (1 decimal)
        'raciones_carbohidratos', # Raciones de carbohidratos por día (entero)
        'dosis_total_insulina',   # Dosis total diaria de insulina (1 decimal)
        'dosis_basal',           # Dosis basal de insulina (1 decimal)
        'dosis_bolo',            # Dosis en bolo de insulina (1 decimal)
        'ratio_insulina_carbohidrato',  # Gramos de CHO cubiertos por 1U insulina (1 decimal)
        'factor_sensibilidad',   # mg/dL reducidos por 1U insulina (1 decimal),
        
        # 5. CLASIFICACIÓN
        'tipo_diabetes_especifico'  # Clasificación según edad de diagnóstico
    ]
    
    df_completo = pd.DataFrame(datos_totales, columns=nombres_columnas)
    

    # VALIDACIÓN Y GUARDADO

    
    print(f"\nDataset generado: {len(df_completo):,} registros")
    print(f"Todos los pacientes tienen diabetes tipo 1 (100%)")
    
    # Guardar el dataset
    df_completo.to_csv(nombre_archivo, index=False, encoding='utf-8')
    print(f"\nDataset guardado como: {nombre_archivo}")
    
    return df_completo

def validar_dataset(df):
    """Valida la coherencia y calidad del dataset generado"""
    
    print("\nVALIDACIÓN DEL DATASET")
    print("="*80)
    
    print(f"\n1. DISTRIBUCIÓN GENERAL:")
    print(f"Total registros: {len(df):,}")
    
    print(f"\n2. DISTRIBUCIÓN POR GÉNERO:")
    print(df['genero'].value_counts())
    
    print(f"\n3. ESTADÍSTICAS ANTROPOMÉTRICAS:")
    print(f"Edad: {df['edad'].min()}-{df['edad'].max()} años (promedio: {df['edad'].mean():.1f})")
    print(f"IMC: {df['imc'].min():.1f}-{df['imc'].max():.1f} (promedio: {df['imc'].mean():.1f})")
    print(f"Normopeso (IMC<25): {(df['imc'] < 25).sum():,} ({(df['imc'] < 25).sum()/len(df)*100:.1f}%)")
    print(f"Sobrepeso (IMC 25-30): {((df['imc'] >= 25) & (df['imc'] < 30)).sum():,} ({((df['imc'] >= 25) & (df['imc'] < 30)).sum()/len(df)*100:.1f}%)")
    print(f"Obesidad (IMC≥30): {(df['imc'] >= 30).sum():,} ({(df['imc'] >= 30).sum()/len(df)*100:.1f}%)")
    
    print(f"\n4. ACTIVIDAD FÍSICA:")
    print(f"Nivel actividad promedio: {df['nivel_actividad_fisica'].mean():.1f}")
    print(f"Horas ejercicio/semana promedio: {df['horas_ejercicio_semana'].mean():.1f}")
    
    print(f"\n5. METABOLISMO GLUCÉMICO:")
    print(f"HbA1c promedio: {df['hba1c'].mean():.1f}%")
    print(f"Glucosa ayunas promedio: {df['glucosa_en_ayunas'].mean():.0f} mg/dL")
    print(f"Glucosa postprandial promedio: {df['glucosa_posprandial'].mean():.0f} mg/dL")
    
    print(f"\n6. INSULINA:")
    print(f"Dosis total promedio: {df['dosis_total_insulina'].mean():.1f} U")
    print(f"Dosis basal promedio: {df['dosis_basal'].mean():.1f} U")
    print(f"Dosis bolo promedio: {df['dosis_bolo'].mean():.1f} U")
    
    print(f"\n7. TIPOS DE DIABETES:")
    tipos = df['tipo_diabetes_especifico'].value_counts()
    for tipo, count in tipos.items():
        print(f"  {tipo}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n8. COHERENCIAS:")
    correlacion_imc_hba1c = df['imc'].corr(df['hba1c'])
    print(f"Correlación IMC-HbA1c: {correlacion_imc_hba1c:.3f}")
    
    correlacion_edad_años_diabetes = df['edad'].corr(df['años_con_diabetes'])
    print(f"Correlación Edad-Años con diabetes: {correlacion_edad_años_diabetes:.3f}")
    
    print(f"\n9. EJEMPLOS DE REGISTROS:")
    print("Primeras 3 filas del dataset:")
    print(df.head(3).to_string())
    
    return df

if __name__ == "__main__":
    try:
        print("GENERADOR DE DATASET DE DIABETES TIPO 1 - 50,000 REGISTROS")

        # Generar dataset principal
        os.makedirs("Base de datos", exist_ok=True)
        df_50k = generar_dataset_antropometrico_coherente("Base de datos/db_diabetes_50k.csv")
        validar_dataset(df_50k)
        
    except Exception as e:
        print(f"Error durante la generación del dataset: {e}")
        import traceback
        traceback.print_exc()