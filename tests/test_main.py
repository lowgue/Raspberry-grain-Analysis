from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

# Mock da inicialização de câmera/threads em ambientes de teste sem hardware
with patch("camera.CameraManager.start", return_value=None):
    from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_set_group():
    # Testa a alteração de grupo/lote ativo
    response = client.post("/api/set_group", json={"group_name": "Teste TDD"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["active_group"] == "Teste TDD"

@patch("main.db_manager")
def test_get_metrics(mock_db):
    # Mock do retorno do banco de dados
    mock_db.get_summary_metrics.return_value = {"healthy": 10, "damaged": 3, "total": 13}
    mock_db.get_group_metrics.return_value = {"Teste TDD": {"healthy": 10, "damaged": 3, "total": 13}}
    
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["healthy"] == 10
    assert data["summary"]["damaged"] == 3
    assert data["summary"]["total"] == 13
    assert data["groups"]["Teste TDD"]["total"] == 13

@patch("main.db_manager")
def test_get_recent_history(mock_db):
    # Mock do histórico recente
    mock_db.get_recent_history.return_value = [
        {"timestamp": "2026-06-03 17:00:00", "status": "damaged", "group_name": "Teste TDD", "confidence": 92.5}
    ]
    response = client.get("/api/recent_history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 1
    assert history[0]["status"] == "damaged"
    assert history[0]["group_name"] == "Teste TDD"
    assert history[0]["confidence"] == 92.5

@patch("main.gpio_controller")
@patch("main.db_manager")
def test_trigger_ejection(mock_db, mock_gpio):
    # Testa o disparo manual do jato de ar
    response = client.post("/api/trigger_ejection", json={"group_name": "Teste TDD"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verifica se os controladores de hardware e banco de dados foram acionados
    mock_gpio.trigger_air_jet.assert_called_once()
    mock_db.log_grain.assert_called_once_with("damaged", "Teste TDD", 1.0)

@patch("main.db_manager")
def test_clear_metrics(mock_db):
    # Testa a limpeza das contagens
    response = client.post("/api/clear_metrics")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_db.clear_metrics.assert_called_once()
