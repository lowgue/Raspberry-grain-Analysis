import logging
import time
import threading

logger = logging.getLogger("gpio_control")

# Tentativa de importar bibliotecas GPIO para o Raspberry Pi
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# Configurações do GPIO
VALVE_PIN = 18  # BCM pin para a solenoide
JET_DURATION_SEC = 0.1  # Duração do jato de ar (100ms)

class GPIOController:
    def __init__(self, pin=VALVE_PIN):
        self.pin = pin
        self.has_gpio = HAS_GPIO
        self._lock = threading.Lock()
        
        if self.has_gpio:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT)
                GPIO.output(self.pin, GPIO.LOW)
                logger.info(f"GPIO inicializado no pino {self.pin} (BCM)")
            except Exception as e:
                logger.error(f"Erro ao configurar GPIO: {e}. Entrando em modo simulado.")
                self.has_gpio = False
        else:
            logger.info("RPi.GPIO não encontrado. Executando em modo SIMULADO.")

    def trigger_air_jet(self):
        """Dispara o jato de ar de forma assíncrona (não-bloqueante)."""
        thread = threading.Thread(target=self._pulse_valve)
        thread.daemon = True
        thread.start()

    def _pulse_valve(self):
        """Executa o pulso elétrico para abrir e fechar a válvula de ar."""
        with self._lock:
            if self.has_gpio:
                try:
                    logger.info("GPIO: [!] JATO DE AR ATIVADO (SINAL HIGH)")
                    GPIO.output(self.pin, GPIO.HIGH)
                    time.sleep(JET_DURATION_SEC)
                    GPIO.output(self.pin, GPIO.LOW)
                    logger.info("GPIO: [.] JATO DE AR DESATIVADO (SINAL LOW)")
                except Exception as e:
                    logger.error(f"Erro ao pulsar GPIO: {e}")
            else:
                logger.info("SIMULAÇÃO: [!] JATO DE AR ATIVADO (100ms)")
                time.sleep(JET_DURATION_SEC)
                logger.info("SIMULAÇÃO: [.] JATO DE AR DESATIVADO")

    def cleanup(self):
        """Limpa as configurações de GPIO."""
        if self.has_gpio:
            try:
                GPIO.cleanup()
                logger.info("GPIO limpo com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao limpar GPIO: {e}")

# Instância única global do controlador de GPIO
gpio_controller = GPIOController()
