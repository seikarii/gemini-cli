import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def execute_python_code_in_partner_workspace(
    code: str, partner_workspace_path: str, timeout: int = 30
) -> dict:
    """
    Ejecuta código Python en un entorno temporal dentro del workspace del compañero.
    Captura stdout y stderr.
    """
    # Asegurarse de que la ruta del workspace del compañero sea absoluta
    if not os.path.isabs(partner_workspace_path):
        logger.error(
            f"partner_workspace_path must be an absolute path: {partner_workspace_path}"
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": "partner_workspace_path must be an absolute path.",
            "return_code": 1,
            "error": "InvalidPath",
        }

    # Crear un archivo temporal para el código dentro del workspace del compañero
    # Esto es para simular la ejecución "en el sandbox del compañero"
    script_path = os.path.join(
        partner_workspace_path, f"temp_exec_script_{os.getpid()}_{time.time_ns()}.py"
    )

    try:
        with open(script_path, "w") as f:
            f.write(code)

        # Ejecutar el script con el intérprete de Python del sistema
        # Considerar usar el python del .venv del compañero si existe y es accesible
        # Para simplificar, usaremos sys.executable por ahora.
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # No lanzar excepción para errores de proceso
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "return_code": 1,
            "error": "TimeoutExpired",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": 1,
            "error": "GeneralException",
        }
    finally:
        # Limpiar el archivo temporal
        if os.path.exists(script_path):
            os.remove(script_path)
