# tests/test_smoke.py
def test_imports():
    import app
    import src.preprocessing
    import src.train_and_select
    # If functions exist
    assert hasattr(src.preprocessing, 'preprocess_data')
