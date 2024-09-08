from os.path import join as os_path_join, abspath as os_path_abspath, dirname as os_path_dirname, exists as os_path_exists
import sys

def get_recurso(relative_path):
    """
    Obtiene la ruta absoluta al recurso, funciona tanto en desarrollo como en PyInstaller
    """
    try:
        # Cuando está empaquetado con PyInstaller
        base_path = sys._MEIPASS
    except AttributeError:
        # Cuando se ejecuta como script
        base_path = os_path_abspath(".")

    # Si lo hay en la carpeta (los tableros) 
    exe_path = os_path_abspath(os_path_join(os_path_dirname(sys.executable), relative_path))
    if os_path_exists(exe_path):
        return exe_path
    else:
        return os_path_join(base_path, relative_path)