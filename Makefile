.PHONY: docs docs-serve

docs:
	sphinx-build docs docs/_build/html

docs-serve:
	python3 -m http.server --directory docs/_build/html
