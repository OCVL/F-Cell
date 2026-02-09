# Per solution at: https://youtrack.jetbrains.com/issue/PY-39748
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    import pkgutil
    pkgutil.extend_path(__path__, __name__)