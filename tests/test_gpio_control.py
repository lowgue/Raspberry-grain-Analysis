import time
import sys
import importlib
from unittest.mock import MagicMock, patch
import pytest

def test_gpio_controller_fallback_simulation():
    # Garante que HAS_GPIO seja False e recarrega o módulo
    with patch("gpio_control.HAS_GPIO", False):
        import gpio_control
        importlib.reload(gpio_control)
        controller = gpio_control.GPIOController(pin=18)
        assert controller.has_gpio is False
        
        # O acionamento em simulação não deve gerar erros e deve rodar de forma assíncrona
        controller.trigger_air_jet()
        time.sleep(0.15)

def test_gpio_controller_real_gpio():
    # Mock do módulo RPi.GPIO
    mock_gpio = MagicMock()
    mock_gpio.OUT = "OUT"
    mock_gpio.HIGH = "HIGH"
    mock_gpio.LOW = "LOW"
    mock_gpio.BCM = "BCM"

    with patch.dict("sys.modules", {"RPi": MagicMock(), "RPi.GPIO": mock_gpio}):
        import gpio_control
        importlib.reload(gpio_control)
        
        # Injeta o mock para simular o hardware ativo
        gpio_control.GPIO = mock_gpio
        gpio_control.HAS_GPIO = True
        
        controller = gpio_control.GPIOController(pin=18)
        assert controller.has_gpio is True
        mock_gpio.setmode.assert_called_once_with(mock_gpio.BCM)
        mock_gpio.setup.assert_called_with(18, mock_gpio.OUT)
        
        # Testa acionamento elétrico simulado via GPIO
        controller.trigger_air_jet()
        time.sleep(0.15)
        mock_gpio.output.assert_any_call(18, mock_gpio.HIGH)
        mock_gpio.output.assert_any_call(18, mock_gpio.LOW)
        
        # Testa limpeza do GPIO
        controller.cleanup()
        mock_gpio.cleanup.assert_called_once()
        
        # Se ocorrer erro na inicialização física, deve reverter para modo simulação
        mock_gpio.setup.side_effect = Exception("Erro físico de permissão")
        controller_fail = gpio_control.GPIOController(pin=18)
        assert controller_fail.has_gpio is False

    # Limpa o estado recarregando o módulo para o estado original
    importlib.reload(gpio_control)
