from ml.class_mapping import to_grain_status

def test_to_grain_status_healthy():
    assert to_grain_status("premium") == "healthy"
    assert to_grain_status("healthy") == "healthy"
    assert to_grain_status("normal") == "healthy"
    assert to_grain_status("good") == "healthy"

def test_to_grain_status_damaged():
    assert to_grain_status("defect") == "damaged"
    assert to_grain_status("damaged") == "damaged"
    assert to_grain_status("black") == "damaged"
    assert to_grain_status("broken") == "damaged"
    assert to_grain_status("fade") == "damaged"
    assert to_grain_status("sour") == "damaged"
    assert to_grain_status("bad") == "damaged"

def test_to_grain_status_fallback():
    # Valores desconhecidos ou vazios
    assert to_grain_status("desconhecido") == "damaged"
    assert to_grain_status("") == "healthy"
    assert to_grain_status(None) == "healthy"
