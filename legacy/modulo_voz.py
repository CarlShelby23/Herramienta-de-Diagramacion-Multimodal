import speech_recognition as sr

def escuchar_comando():
    reconocedor = sr.Recognizer()
    
    with sr.Microphone() as origen:
        print("Ajustando ruido de fondo... (silencio breve)")
        reconocedor.adjust_for_ambient_noise(origen, duration=1)
        print("Escuchando comando (Ej. 'Crear nodo', 'Conectar')...")
        
        try:
            audio = reconocedor.listen(origen, timeout=5, phrase_time_limit=3)
            print("Procesando audio...")
            # Uso del motor gratuito de Google para el MVP
            comando = reconocedor.recognize_google(audio, language="es-MX")
            print(f">> Comando reconocido: '{comando}'")
            return comando.lower()
        
        except sr.WaitTimeoutError:
            print("Tiempo de espera agotado. No se detectó voz.")
        except sr.UnknownValueError:
            print("No se pudo interpretar el audio.")
        except sr.RequestError as e:
            print(f"Error de conexión con el servicio ASR: {e}")
            
    return None

if __name__ == "__main__":
    escuchar_comando()