import tkinter as tk
import cv2
import threading
import time
import json
from modulo_vision import GestorVisionMultimodal
from modulo_voz import escuchar_comando

class AplicacionDiagramacion:
# 1. Agrega 'sesion_id' y 'complejidad' como parámetros
    def __init__(self, root, sesion_id="TEST-001", complejidad="Media"):
        self.root = root
        self.root.title(f"Herramienta de Diagramación Multimodal | Sesión: {sesion_id} [{complejidad}]")
        
        # Lienzo minimalista (Carga cognitiva baja)
        self.canvas = tk.Canvas(root, width=1280, height=720, bg="#F4F5F7", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.gestor_vision = GestorVisionMultimodal()
        self.cap = cv2.VideoCapture(0)
        
        # Almacenamiento de estado y telemetría
        self.nodos = []
        self.enlaces = []
        self.log_telemetria = []
        self.comando_actual = None
        
        # 2. Asigna los valores recibidos a las variables de tu clase (para el JSON R-09)
        self.sesion_id = sesion_id
        self.complejidad = complejidad
        
        # Variables de Edición Topológica (Arrastre y Conexión)
        self.nodo_activo = None
        self.nodo_arrastrado = None
        self.linea_previsualizacion = None

        # UI Elements
        self.id_cursor = self.canvas.create_oval(-10, -10, -10, -10, fill="#64748B", outline="")
        self.texto_hud = self.canvas.create_text(640, 680, text=f"Iniciando motores multimodales... | {self.sesion_id}", font=("Arial", 14, "bold"), fill="#1E293B")
        
        # Menú contextual de contingencia (R-04)
        self.menu_contextual = []

        # Hilo de voz no bloqueante
        self.hilo_voz = threading.Thread(target=self.bucle_escucha_voz, daemon=True)
        self.hilo_voz.start()

        # Iniciar bucle principal
        self.actualizar_interfaz()

    def bucle_escucha_voz(self):
        # Bucle en segundo plano para ASR.
        while True:
            self.canvas.itemconfig(self.texto_hud, text="🎙️ Escuchando comandos...", fill="#1E293B")
            comando = escuchar_comando()
            if comando:
                self.comando_actual = comando
                self.canvas.itemconfig(self.texto_hud, text=f"Comando detectado: '{comando}'", fill="#10B981")
            time.sleep(0.1)

    def obtener_nodo_en_coordenada(self, x, y):
        # Devuelve el nodo que colisiona con el cursor.
        margen = 30
        for nodo in reversed(self.nodos):
            if abs(nodo["x"] - x) < 90 + margen and abs(nodo["y"] - y) < 30 + margen:
                return nodo
        return None

    def fusion_multimodal(self, telemetria_gesto):
        # Fusión a Nivel de Decisión: Cruza voz y gesto.
        x, y = telemetria_gesto["cursor_x"], telemetria_gesto["cursor_y"]
        estado_gestual = telemetria_gesto["estado_gesto"]
        comando = self.comando_actual.lower() if self.comando_actual else ""

        # Sinergia 1: Creación
        if "crear" in comando and estado_gestual == "CURSOR_MOVE" and x and y:
            palabras = comando.split("nodo")
            concepto = palabras[1].strip() if len(palabras) > 1 else "Nuevo Concepto"
            self.crear_nodo(x, y, concepto)
            self.registrar_evento("NODE_CREATE", comando, x, y, modalidad="MULTIMODAL")
            self.comando_actual = None

        # Sinergia 2: Conexión (Requiere que un nodo esté siendo arrastrado)
        if "conectar" in comando and self.nodo_arrastrado and x and y:
            nodo_destino = self.obtener_nodo_en_coordenada(x, y)
            if nodo_destino and nodo_destino != self.nodo_arrastrado:
                self.conectar_nodos(self.nodo_arrastrado, nodo_destino)
                self.registrar_evento("EDGE_CREATE", comando, x, y, modalidad="MULTIMODAL")
            else:
                self.registrar_evento("TASK_INTERRUPTION", "Intento de conexión fallido", x, y, modalidad="VOZ")
            
            # Limpiar estado de conexión
            self.nodo_arrastrado = None
            if self.linea_previsualizacion:
                self.canvas.delete(self.linea_previsualizacion)
            self.comando_actual = None

    def crear_nodo(self, x, y, texto):
        # Renderiza nodo geométrico.
        ancho, alto = 180, 60
        x1, y1 = x - ancho//2, y - alto//2
        x2, y2 = x + ancho//2, y + alto//2
        
        id_rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="#FFFFFF", outline="#64748B", width=2)
        id_texto = self.canvas.create_text(x, y, text=texto.capitalize(), font=("Arial", 12, "bold"), fill="#1E293B")
        
        self.nodos.append({"id_rect": id_rect, "id_texto": id_texto, "x": x, "y": y, "texto": texto})

    def conectar_nodos(self, nodo_origen, nodo_destino):
        # Genera arista visual entre nodos.
        id_linea = self.canvas.create_line(nodo_origen["x"], nodo_origen["y"], nodo_destino["x"], nodo_destino["y"], fill="#2563EB", width=3)
        self.canvas.tag_lower(id_linea) # Enviar al fondo
        self.enlaces.append({"origen": nodo_origen, "destino": nodo_destino, "id_linea": id_linea})

    def mostrar_menu_contextual(self, nodo):
        # Muestra botones de contingencia (R-04) sobre el nodo.
        self.ocultar_menu_contextual()
        x, y = nodo["x"], nodo["y"] - 50
        id_bg = self.canvas.create_rectangle(x-90, y-15, x+90, y+15, fill="#FFFFFF", outline="#94A3B8")
        id_txt = self.canvas.create_text(x, y, text="[Conectar]   [Borrar]   [Deshacer]", font=("Arial", 9, "bold"), fill="#1E293B")
        self.menu_contextual = [id_bg, id_txt]

    def ocultar_menu_contextual(self):
        for item in self.menu_contextual:
            self.canvas.delete(item)
        self.menu_contextual = []

    def actualizar_interfaz(self):
        # Bucle principal asíncrono (Visión + UI).
        exito, fotograma = self.cap.read()
        if exito:
            fotograma_procesado, telemetria_gesto = self.gestor_vision.procesar_fotograma(fotograma)

            cv2.imshow("Camara - Motor de Vision", fotograma_procesado)
            cv2.waitKey(1) 

            if telemetria_gesto["valido"] and telemetria_gesto["cursor_x"] is not None:
                x, y = telemetria_gesto["cursor_x"], telemetria_gesto["cursor_y"]
                estado = telemetria_gesto["estado_gesto"]
                
                # Actualizar posición y color del cursor (Feedback R-20)
                color_cursor = "#64748B"
                if estado == "PELLIZCO_INICIADO": color_cursor = "#3B82F6" 
                elif estado == "NODE_SELECT": color_cursor = "#10B981" 
                elif estado == "NODE_DRAG": color_cursor = "#F59E0B"
                
                self.canvas.coords(self.id_cursor, x - 10, y - 10, x + 10, y + 10)
                self.canvas.itemconfig(self.id_cursor, fill=color_cursor)
                self.canvas.tag_raise(self.id_cursor)

                # Lógica espacial: Selección, Menú Contextual y Arrastre
                nodo_bajo_cursor = self.obtener_nodo_en_coordenada(x, y)

                if estado == "NODE_SELECT":
                    if nodo_bajo_cursor:
                        self.nodo_activo = nodo_bajo_cursor
                        self.canvas.itemconfig(self.nodo_activo["id_rect"], outline="#2563EB", fill="#EFF6FF")
                        self.mostrar_menu_contextual(self.nodo_activo)
                    else:
                        if self.nodo_activo:
                            self.canvas.itemconfig(self.nodo_activo["id_rect"], outline="#64748B", fill="#FFFFFF")
                        self.nodo_activo = None
                        self.ocultar_menu_contextual()

                elif estado == "NODE_DRAG": 
                    # Simular "Puño" asumiendo que el estado es provisto por modulo_vision
                    if not self.nodo_arrastrado and nodo_bajo_cursor:
                        self.nodo_arrastrado = nodo_bajo_cursor
                    
                    if self.nodo_arrastrado:
                        # Previsualizar línea elástica de conexión
                        if self.linea_previsualizacion:
                            self.canvas.coords(self.linea_previsualizacion, self.nodo_arrastrado["x"], self.nodo_arrastrado["y"], x, y)
                        else:
                            self.linea_previsualizacion = self.canvas.create_line(self.nodo_arrastrado["x"], self.nodo_arrastrado["y"], x, y, fill="#3B82F6", dash=(4, 4), width=2)

                elif estado == "CURSOR_MOVE" and self.nodo_arrastrado:
                    # Soltar nodo (Drop) sin el comando verbal (Interrupción)
                    self.registrar_evento("TASK_INTERRUPTION", "Arrastre abandonado", x, y, modalidad="GESTO")
                    if self.linea_previsualizacion:
                        self.canvas.delete(self.linea_previsualizacion)
                        self.linea_previsualizacion = None
                    self.nodo_arrastrado = None

                # Ejecutar fusión multimodal
                self.fusion_multimodal(telemetria_gesto)

        self.root.after(15, self.actualizar_interfaz)

    def registrar_evento(self, tipo_evento, comando_voz, x, y, modalidad="SISTEMA"):
        # Persistencia JSON atómica (R-05).
        evento = {
            "timestamp_ms": int(time.time() * 1000),
            "modalidad": modalidad,
            "evento": tipo_evento,
            "comando_voz": comando_voz,
            "coordenada_x": x,
            "coordenada_y": y
        }
        self.log_telemetria.append(evento)
        
        with open(f"{self.sesion_id}_log.json", "w") as f:
            json.dump(self.log_telemetria, f, indent=4)
        print(f"[{evento['timestamp_ms']}] Registrado: {tipo_evento} | Modalidad: {modalidad}")

if __name__ == "__main__":
    ventana = tk.Tk()
    ventana.geometry("1280x720")
    app = AplicacionDiagramacion(ventana)
    ventana.mainloop()