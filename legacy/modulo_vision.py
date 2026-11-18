import cv2
import mediapipe as mp
import time
import math

class GestorVisionMultimodal:
    def __init__(self, umbral_pellizco_px=30, umbral_tiempo_ms=250):
        self.mp_manos = mp.solutions.hands
        self.manos = self.mp_manos.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_dibujo = mp.solutions.drawing_utils
        
        # Parámetros de Riesgo R-20
        self.umbral_pellizco_px = umbral_pellizco_px
        self.umbral_tiempo_ms = umbral_tiempo_ms / 1000.0  
        
        self.estado_actual = "REPOSO"
        self.tiempo_inicio_gesto = 0
        self.gesto_confirmado = False
        
        # Suavizado R-20
        self.cursor_x_prev = 0
        self.cursor_y_prev = 0

    def calcular_distancia(self, p1, p2, width, height):
        x1, y1 = int(p1.x * width), int(p1.y * height)
        x2, y2 = int(p2.x * width), int(p2.y * height)
        return math.hypot(x2 - x1, y2 - y1)

    def suavizar_coordenada(self, nuevo_val, val_previo, factor_suavizado=0.6):
        if val_previo == 0: return nuevo_val
        return int((nuevo_val * factor_suavizado) + (val_previo * (1.0 - factor_suavizado)))

    def procesar_fotograma(self, fotograma):
        fotograma = cv2.flip(fotograma, 1)
        h, w, c = fotograma.shape
        fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
        
        resultados = self.manos.process(fotograma_rgb)
        
        paquete_telemetria = {
            "timestamp_ms": int(time.time() * 1000),
            "estado_gesto": "REPOSO",
            "cursor_x": None,
            "cursor_y": None,
            "valido": False
        }

        if resultados.multi_hand_landmarks:
            for puntos_mano in resultados.multi_hand_landmarks:
                self.mp_dibujo.draw_landmarks(fotograma, puntos_mano, self.mp_manos.HAND_CONNECTIONS)
                
                x_indice = int(puntos_mano.landmark[8].x * w)
                y_indice = int(puntos_mano.landmark[8].y * h)
                
                cursor_x = self.suavizar_coordenada(x_indice, self.cursor_x_prev)
                cursor_y = self.suavizar_coordenada(y_indice, self.cursor_y_prev)
                self.cursor_x_prev, self.cursor_y_prev = cursor_x, cursor_y
                
                paquete_telemetria["cursor_x"] = cursor_x
                paquete_telemetria["cursor_y"] = cursor_y
                
                dist_pellizco = self.calcular_distancia(puntos_mano.landmark[4], puntos_mano.landmark[8], w, h)
                
                if dist_pellizco < self.umbral_pellizco_px:
                    if self.estado_actual != "PELLIZCO_INICIADO":
                        self.estado_actual = "PELLIZCO_INICIADO"
                        self.tiempo_inicio_gesto = time.time()
                    
                    tiempo_transcurrido = time.time() - self.tiempo_inicio_gesto
                    if tiempo_transcurrido >= self.umbral_tiempo_ms:
                        paquete_telemetria["estado_gesto"] = "NODE_SELECT"
                        paquete_telemetria["valido"] = True
                        cv2.circle(fotograma, (cursor_x, cursor_y), 15, (0, 255, 0), cv2.FILLED) 
                    else:
                        cv2.circle(fotograma, (cursor_x, cursor_y), 15, (0, 165, 255), 2) 
                else:
                    self.estado_actual = "REPOSO"
                    self.tiempo_inicio_gesto = 0
                    paquete_telemetria["estado_gesto"] = "CURSOR_MOVE"
                    paquete_telemetria["valido"] = True
                    cv2.circle(fotograma, (cursor_x, cursor_y), 10, (255, 0, 0), cv2.FILLED) 
                    
        return fotograma, paquete_telemetria

if __name__ == "__main__":
    gestor = GestorVisionMultimodal()
    cap = cv2.VideoCapture(0)
    
    print("Iniciando cámara Logitech Brio... presiona ESC para salir.")
    while cap.isOpened():
        exito, frame = cap.read()
        if not exito: break
        
        frame_procesado, telemetria = gestor.procesar_fotograma(frame)
        
        if telemetria["valido"] and telemetria["estado_gesto"] == "NODE_SELECT":
            print(f"[{telemetria['timestamp_ms']}] Gesto Confirmado: {telemetria['estado_gesto']} en X:{telemetria['cursor_x']} Y:{telemetria['cursor_y']}")
            
        cv2.imshow('Rastreo Multimodal (R-20)', frame_procesado)
        if cv2.waitKey(5) & 0xFF == 27: break
        
    cap.release()
    cv2.destroyAllWindows()