from os.path import join as os_path_join, abspath as os_path_abspath
import sys
def get_recurso(relative_path):
    """Obtén la ruta absoluta al recurso, funciona tanto en desarrollo como en PyInstaller"""
    try:
        # Cuando está empaquetado con PyInstaller
        base_path = sys._MEIPASS
    except AttributeError:
        # Cuando se ejecuta como script
        base_path = os_path_abspath(".")

    return f"{os_path_join(base_path, relative_path)}"
