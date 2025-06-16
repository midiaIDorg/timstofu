make:
    echo "Welcome to Project 'timstofu'"

upload_test_pypi:
    twine check dist/*
    python -m pip install --upgrade twine
    twine upload --repository testpypi dist/*

upload_pypi:
    twine check dist/*
    python -m pip install --upgrade twine
    twine upload dist/* 

ve_timstofu:
    python3 -m venv ve_timstofu
