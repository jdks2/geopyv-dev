project = "geopyv-dev"
copyright = "2025"
author = "Sam Stanier & Jonathan Smith"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "undoc-members": True,
}

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "scripts"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/jdks2/geopyv-dev",
}
