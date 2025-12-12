# simulador_diabetes_rl.py
import numpy as np
import random

class SimuladorDiabetesRL:
    """
    Simulador simplificado para entrenamiento RL
    Basado en la propuesta del PDF
    """
    
    def __init__(self, paciente_id=1, tipo_sensibilidad="normal"):
        """
        Args:
            paciente_id: Identificador del paciente
            tipo_sensibilidad: "baja", "normal", "alta"
        """
        self.paciente_id = paciente_id
        self.tipo_sensibilidad = tipo_sensibilidad
        
        # ESPACIO DE ESTADOS según PDF
        self.categorias_glucosa = ["<70", "70-180", "180-250", ">250"]
        self.categorias_insulina_activa = [0, 5, 10, 15, 20]
        self.categorias_tiempo_dosis = ["0-60", "60-120", "120-240", ">240"]
        self.categorias_sensibilidad = ["baja", "normal", "alta"]
        
        # ACCIONES según PDF
        self.acciones = [0, 5, 10, 15]  # unidades de insulina
        
        # Parámetros según sensibilidad
        self._configurar_parametros()
        
        # Inicializar
        self.reset()
    
    def _configurar_parametros(self):
        """Configura parámetros según tipo de paciente"""
        # Efecto de insulina por unidad (mg/dL reducidos por unidad)
        if self.tipo_sensibilidad == "baja":
            self.efecto_insulina = 1.5
            self.metabolismo_basal = 2.0
        elif self.tipo_sensibilidad == "alta":
            self.efecto_insulina = 3.5
            self.metabolismo_basal = 2.5
        else:  # normal
            self.efecto_insulina = 2.5
            self.metabolismo_basal = 2.2
        
        # Comidas simuladas (picos de glucosa)
        self.horarios_comida = [8*60, 13*60, 20*60]  # 8am, 1pm, 8pm
        self.fuerza_comida = 25  # mg/dL por comida
    
    def reset(self):
        """Reinicia la simulación"""
        # Estado inicial aleatorio pero realista
        self.glucosa = np.random.uniform(80, 160)
        self.insulina_activa = 0
        self.tiempo_desde_dosis = np.random.randint(120, 300)  # 2-5 horas
        self.tiempo_actual = 0  # minutos desde inicio
        self.pasos = 0
        
        return self._discretizar_estado()
    
    def _discretizar_estado(self):
        """Convierte estado continuo a discreto según PDF"""
        # 1. Glucosa
        if self.glucosa < 70:
            glucosa_cat = "<70"
        elif self.glucosa <= 180:
            glucosa_cat = "70-180"
        elif self.glucosa <= 250:
            glucosa_cat = "180-250"
        else:
            glucosa_cat = ">250"
        
        # 2. Insulina activa (redondeada a múltiplos de 5)
        insulina_cat = min(20, round(self.insulina_activa / 5) * 5)
        
        # 3. Tiempo desde última dosis
        if self.tiempo_desde_dosis <= 60:
            tiempo_cat = "0-60"
        elif self.tiempo_desde_dosis <= 120:
            tiempo_cat = "60-120"
        elif self.tiempo_desde_dosis <= 240:
            tiempo_cat = "120-240"
        else:
            tiempo_cat = ">240"
        
        # 4. Sensibilidad (fija por paciente)
        sensibilidad_cat = self.tipo_sensibilidad
        
        return (glucosa_cat, insulina_cat, tiempo_cat, sensibilidad_cat)
    
    def _estado_a_indice(self, estado):
        """Convierte estado discreto a índice para Q-table"""
        glucosa_cat, insulina_cat, tiempo_cat, sensibilidad_cat = estado
        
        # Mapear categorías a índices
        g_idx = self.categorias_glucosa.index(glucosa_cat)
        i_idx = self.categorias_insulina_activa.index(insulina_cat)
        t_idx = self.categorias_tiempo_dosis.index(tiempo_cat)
        s_idx = self.categorias_sensibilidad.index(sensibilidad_cat)
        
        # Índice único: g*60 + i*12 + t*3 + s
        # (4*5*4*3 = 240 estados totales)
        return g_idx*60 + i_idx*12 + t_idx*3 + s_idx
    
    def _calcular_efecto_comida(self):
        """Calcula efecto de comidas en la glucosa"""
        efecto = 0
        for hora_comida in self.horarios_comida:
            tiempo_desde_comida = self.tiempo_actual - hora_comida
            # Comida afecta entre 0-90 minutos después
            if 0 <= tiempo_desde_comida <= 90:
                # Pico a los 60 minutos
                if tiempo_desde_comida <= 60:
                    progresion = tiempo_desde_comida / 60
                else:
                    progresion = 1 - ((tiempo_desde_comida - 60) / 30)
                efecto += self.fuerza_comida * progresion
        
        return efecto
    
    def step(self, accion):
        """
        Ejecuta un paso de simulación (30 minutos)
        
        Args:
            accion: Índice de la acción (0-3)
        
        Returns:
            (nuevo_estado, recompensa, terminado)
        """
        # Convertir índice a dosis
        dosis = self.acciones[accion]
        
        # 1. ADMINISTRAR INSULINA
        if dosis > 0:
            self.insulina_activa += dosis
            self.tiempo_desde_dosis = 0
        
        # 2. SIMULAR FISIOLOGÍA
        # Efecto insulina
        reduccion_glucosa = self.insulina_activa * self.efecto_insulina
        
        # Metabolismo basal
        aumento_glucosa = self.metabolismo_basal
        
        # Efecto comidas
        efecto_comida = self._calcular_efecto_comida()
        
        # Variabilidad aleatoria
        ruido = np.random.normal(0, 5)
        
        # Cambio total en glucosa
        delta_glucosa = aumento_glucosa + efecto_comida - reduccion_glucosa + ruido
        
        # 3. ACTUALIZAR VARIABLES
        self.glucosa += delta_glucosa
        self.glucosa = max(40, min(400, self.glucosa))  # Límites seguros
        
        # Decaer insulina (25% cada 30min)
        self.insulina_activa *= 0.75
        if self.insulina_activa < 0.5:
            self.insulina_activa = 0
        
        # Actualizar tiempos
        self.tiempo_desde_dosis += 30
        self.tiempo_actual += 30
        self.pasos += 1
        
        # 4. CALCULAR RECOMPENSA (EXACTA del PDF)
        recompensa = self._calcular_recompensa(dosis)
        
        # 5. VERIFICAR TERMINACIÓN
        terminado = self.pasos >= 48  # 24 horas (48 pasos de 30min)
        
        # 6. NUEVO ESTADO
        nuevo_estado = self._discretizar_estado()
        
        return nuevo_estado, recompensa, terminado
    
    def _calcular_recompensa(self, dosis):
        """Calcula recompensa según PDF"""
        # Rango ideal
        if 70 <= self.glucosa <= 180:
            return 10
        
        # Hipoglucemia severa
        elif self.glucosa < 60:
            return -20
        
        # Hiperglucemia severa
        elif self.glucosa > 260:
            return -15
        
        # Dosis excesiva
        elif dosis > 20:
            return -5
        
        # Penalización leve por otros estados
        else:
            return -1
    
    def get_estado_actual(self):
        """Retorna estado actual discreto"""
        return self._discretizar_estado()
    
    def get_info(self):
        """Retorna información del estado actual"""
        return {
            'glucosa': round(self.glucosa, 1),
            'insulina_activa': round(self.insulina_activa, 1),
            'tiempo_desde_dosis': self.tiempo_desde_dosis,
            'hora': f"{self.tiempo_actual//60:02d}:{(self.tiempo_actual%60):02d}",
            'pasos': self.pasos,
            'paciente_id': self.paciente_id,
            'sensibilidad': self.tipo_sensibilidad
        }