"""Mapeamento de classes do Coffee Bean Dataset para o sistema (healthy/damaged)."""

# CoffeeBeansGradingV3 (Roboflow): defect, premium
# Outros datasets de grãos verdes: black, broken, fade, sour -> damaged; normal -> healthy
COFFEE_CLASS_TO_STATUS = {
    "defect": "damaged",
    "premium": "healthy",
    "damaged": "damaged",
    "healthy": "healthy",
    "black": "damaged",
    "broken": "damaged",
    "fade": "damaged",
    "sour": "damaged",
    "normal": "healthy",
    "good": "healthy",
    "bad": "damaged",
}


def to_grain_status(class_name: str) -> str:
    """Converte nome da classe YOLO para 'healthy' ou 'damaged'."""
    key = (class_name or "").strip().lower()
    return COFFEE_CLASS_TO_STATUS.get(key, "damaged" if key else "healthy")
