import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from interfaz_lienzo import AplicacionDiagramacion # Importamos el lienzo que ya hicimos

class DashboardInicio:
    def __init__(self, root):
        self.root = root
        self.root.title("Configuración Pre-Sesión - TDAH Multimodal")
        self.root.geometry("800x500")
        self.root.configure(bg="#F4F5F7")

        # --- Contenedor Principal ---
        marco_principal = tk.Frame(self.root, bg="#FFFFFF", padx=30, pady=30, relief="solid", bd=1)
        marco_principal.pack(expand=True, fill="both", padx=40, pady=40)

        # Título
        tk.Label(marco_principal, text="Configuración de Sesión Experimental", font=("Arial", 16, "bold"), bg="#FFFFFF", fg="#1E293B").pack(anchor="w", pady=(0, 10))
        tk.Label(marco_principal, text="Complete los datos del participante para anonimizar el registro JSON.", font=("Arial", 10), bg="#FFFFFF", fg="#64748B").pack(anchor="w", pady=(0, 20))

        # --- 1. Código de Participante (Anonimato R-09) ---
        tk.Label(marco_principal, text="1. Código de Participante (Anónimo):", font=("Arial", 10, "bold"), bg="#FFFFFF", fg="#334155").pack(anchor="w")
        self.entrada_codigo = tk.Entry(marco_principal, font=("Consolas", 12), width=30, bg="#F8FAFC", relief="solid")
        self.entrada_codigo.insert(0, "PART-2026-")
        self.entrada_codigo.pack(anchor="w", pady=(5, 20))

        # --- 2. Selección de Complejidad (R-07) ---
        tk.Label(marco_principal, text="2. Condición / Complejidad de Tarea:", font=("Arial", 10, "bold"), bg="#FFFFFF", fg="#334155").pack(anchor="w")
        self.var_complejidad = tk.StringVar(value="Media (Parcial)")
        
        marco_opciones = tk.Frame(marco_principal, bg="#FFFFFF")
        marco_opciones.pack(anchor="w", pady=(5, 20))
        
        opciones = ["Familiarización", "Baja (Guiada)", "Media (Parcial)", "Alta (Abierta)"]
        for i, opc in enumerate(opciones):
            ttk.Radiobutton(marco_opciones, text=opc, variable=self.var_complejidad, value=opc).grid(row=i//2, column=i%2, padx=10, pady=5, sticky="w")

        # --- Verificación de Hardware (R-12, R-18) ---
        tk.Label(marco_principal, text="3. Verificación de Periféricos:", font=("Arial", 10, "bold"), bg="#FFFFFF", fg="#334155").pack(anchor="w")
        self.lbl_estado_cam = tk.Label(marco_principal, text="Verificando cámara...", fg="#F59E0B", bg="#FFFFFF", font=("Arial", 9))
        self.lbl_estado_cam.pack(anchor="w", pady=2)
        
        # Botón de Inicio
        btn_iniciar = tk.Button(marco_principal, text="INICIAR SESIÓN Y TELEMETRÍA JSON >>", font=("Arial", 11, "bold"), bg="#1E293B", fg="white", padx=20, pady=10, command=self.lanzar_lienzo)
        btn_iniciar.pack(side="bottom", pady=20)

        # Iniciar verificación de cámara en background
        self.root.after(500, self.verificar_camara)

    def verificar_camara(self):
        #Intenta abrir la cámara web brevemente para asegurar que no hay errores antes de la prueba.
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self.lbl_estado_cam.config(text="Cámara (Logitech Brio): OK", fg="#10B981")
            cap.release()
        else:
            self.lbl_estado_cam.config(text="Error: No se detecta cámara web", fg="#EF4444")
            messagebox.showerror("Error de Hardware", "No se detecta la cámara web. Conecte el dispositivo para continuar.")

    def lanzar_lienzo(self):
        codigo = self.entrada_codigo.get().strip()
        complejidad = self.var_complejidad.get()

        if len(codigo) < 5:
            messagebox.showwarning("Atención", "Ingrese un código de participante válido.")
            return

        # Cierra el dashboard
        self.root.destroy()

        # --- ABRE EL LIENZO PRINCIPAL ---
        ventana_lienzo = tk.Tk()
        ventana_lienzo.geometry("1280x720")
        
        # Pasamos los parámetros al lienzo
        app = AplicacionDiagramacion(ventana_lienzo, sesion_id=codigo, complejidad=complejidad)
        ventana_lienzo.mainloop()

if __name__ == "__main__":
    ventana_principal = tk.Tk()
    app_inicio = DashboardInicio(ventana_principal)
    ventana_principal.mainloop()