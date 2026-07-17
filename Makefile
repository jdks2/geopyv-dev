.PHONY: tutorials docs-assets docs docs-serve

tutorials:
	python3 geopyv_dev/tutorials/make_tutorials.py

docs-assets:
	python3 docs/build_assets.py

docs: docs-assets
	jupyter-book build docs/

docs-serve:
	python3 -m http.server --directory docs/_build/html
