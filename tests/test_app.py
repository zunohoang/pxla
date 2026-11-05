"""Exercise sample analysis and missing-model behavior without a browser."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_app_initial_and_coffee_empty():
    app = AppTest.from_file(Path('app.py').resolve()).run(timeout=30)
    assert not app.exception
    app.sidebar.selectbox[0].select('coffee').run()
    assert not app.exception
    if not Path('models/coffee_knn.npz').is_file():
        assert any('Chưa có mô hình' in item.value for item in app.sidebar.info)


@pytest.mark.skipif(not list(Path('data/raw/rice').glob('*/*')), reason='Dataset not downloaded')
def test_app_sample():
    app = AppTest.from_file(Path('app.py').resolve()).run(timeout=30)
    app.radio[0].set_value('Ảnh mẫu').run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
