import sqlite3
import os
import queue
import threading
import logging
import time
from datetime import datetime

logger = logging.getLogger("database")

DB_PATH = os.path.join(os.path.dirname(__file__), "grain_metrics.db")

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.write_queue = queue.Queue()
        self._init_db()
        
        # Inicia a thread de escrita assíncrona
        self.writer_thread = threading.Thread(target=self._process_write_queue, daemon=True)
        self.writer_thread.start()
        logger.info("Banco de dados SQLite inicializado e thread de escrita iniciada.")

    def _init_db(self):
        """Inicializa o banco de dados e cria as tabelas necessárias se não existirem."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,      -- 'healthy' (saudável) ou 'damaged' (estragado)
                group_name TEXT NOT NULL,  -- Grupo específico (ex: 'Grupo A', 'Grupo B')
                confidence REAL NOT NULL   -- Grau de certeza da IA
            )
        """)
        # Criação de índices para consultas rápidas nos gráficos
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON grain_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON grain_events(status)")
        conn.commit()
        conn.close()

    def _process_write_queue(self):
        """Thread worker para processar inserções de forma não-bloqueante."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        while True:
            try:
                # Espera por um item para inserir
                status, group_name, confidence = self.write_queue.get()
                cursor.execute(
                    "INSERT INTO grain_events (status, group_name, confidence) VALUES (?, ?, ?)",
                    (status, group_name, confidence)
                )
                conn.commit()
                self.write_queue.task_done()
            except Exception as e:
                logger.error(f"Erro ao salvar evento de grão no banco: {e}")
                time.sleep(1)  # Evita loop infinito rápido em caso de erro persistente

    def log_grain(self, status: str, group_name: str, confidence: float):
        """Enfileira um evento de classificação de grão para escrita assíncrona."""
        self.write_queue.put((status, group_name, confidence))

    def get_summary_metrics(self):
        """Retorna contadores totais de grãos por status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT status, COUNT(*) FROM grain_events GROUP BY status")
            rows = cursor.fetchall()
            metrics = {"healthy": 0, "damaged": 0, "total": 0}
            for status, count in rows:
                if status in metrics:
                    metrics[status] = count
                metrics["total"] += count
            return metrics
        except Exception as e:
            logger.error(f"Erro ao ler métricas de resumo: {e}")
            return {"healthy": 0, "damaged": 0, "total": 0}
        finally:
            conn.close()

    def get_group_metrics(self):
        """Retorna a contagem agrupada por status e grupo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT group_name, status, COUNT(*) FROM grain_events GROUP BY group_name, status")
            rows = cursor.fetchall()
            groups = {}
            for group, status, count in rows:
                if group not in groups:
                    groups[group] = {"healthy": 0, "damaged": 0, "total": 0}
                groups[group][status] = count
                groups[group]["total"] += count
            return groups
        except Exception as e:
            logger.error(f"Erro ao ler métricas de grupo: {e}")
            return {}
        finally:
            conn.close()

    def get_recent_history(self, limit=10):
        """Retorna os últimos eventos de classificação."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT timestamp, status, group_name, confidence FROM grain_events ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "timestamp": row[0],
                    "status": row[1],
                    "group_name": row[2],
                    "confidence": round(row[3] * 100, 1)
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Erro ao ler histórico recente: {e}")
            return []
        finally:
            conn.close()

    def clear_metrics(self):
        """Limpa todo o histórico de contagem de grãos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM grain_events")
            conn.commit()
            logger.info("Banco de dados limpo com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao limpar banco de dados: {e}")
        finally:
            conn.close()

# Instância única global do banco de dados
db_manager = DatabaseManager()
