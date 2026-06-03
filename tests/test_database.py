import os
import pytest
from database import DatabaseManager

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_grain_metrics.db"
    manager = DatabaseManager(db_path=str(db_file))
    yield manager
    # Aguarda a fila de escrita assíncrona ser totalmente processada
    manager.write_queue.join()

def test_database_initialization(test_db):
    assert os.path.exists(test_db.db_path)
    
    # Verifica resumo inicial
    summary = test_db.get_summary_metrics()
    assert summary["healthy"] == 0
    assert summary["damaged"] == 0
    assert summary["total"] == 0

def test_log_grain_and_metrics(test_db):
    # Loga grãos saudáveis e danificados
    test_db.log_grain("healthy", "Lote 1", 0.95)
    test_db.log_grain("healthy", "Lote 1", 0.98)
    test_db.log_grain("damaged", "Lote 1", 0.85)
    test_db.log_grain("damaged", "Lote 2", 0.90)

    # Espera esvaziar a fila
    test_db.write_queue.join()

    # Verifica resumo total
    summary = test_db.get_summary_metrics()
    assert summary["healthy"] == 2
    assert summary["damaged"] == 2
    assert summary["total"] == 4

    # Verifica métricas por lote/grupo
    groups = test_db.get_group_metrics()
    assert "Lote 1" in groups
    assert groups["Lote 1"]["healthy"] == 2
    assert groups["Lote 1"]["damaged"] == 1
    assert groups["Lote 1"]["total"] == 3

    assert "Lote 2" in groups
    assert groups["Lote 2"]["healthy"] == 0
    assert groups["Lote 2"]["damaged"] == 1
    assert groups["Lote 2"]["total"] == 1

def test_get_recent_history(test_db):
    test_db.log_grain("healthy", "Lote 1", 0.95)
    test_db.log_grain("damaged", "Lote 2", 0.80)
    test_db.write_queue.join()

    history = test_db.get_recent_history(limit=5)
    assert len(history) == 2
    assert history[0]["status"] == "damaged"
    assert history[0]["group_name"] == "Lote 2"
    assert history[0]["confidence"] == 80.0  # Retorna percentual
    assert history[1]["status"] == "healthy"
    assert history[1]["group_name"] == "Lote 1"
    assert history[1]["confidence"] == 95.0

def test_clear_metrics(test_db):
    test_db.log_grain("healthy", "Lote 1", 0.95)
    test_db.write_queue.join()

    # Confirma que há dados
    assert test_db.get_summary_metrics()["total"] == 1

    # Limpa o banco
    test_db.clear_metrics()

    # Confirma que foi limpo
    assert test_db.get_summary_metrics()["total"] == 0
