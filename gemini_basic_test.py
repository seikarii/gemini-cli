
'''
Este script prueba las funciones básicas de la API de Gemini.
'''
import os

# Suponiendo que las herramientas están en un módulo accesible.
# Si se ejecutan en el entorno de Gemini, esto podría gestionarse de forma diferente.

# Para simular el entorno, definimos funciones ficticias que llaman a las herramientas de Gemini.
# En un entorno real, estas llamadas serían directas.
def write_file(file_path, content):
    from core.tools.local_file_system import write_file as write_file_tool
    return write_file_tool.execute(file_path=file_path, content=content)

def read_file(absolute_path):
    from core.tools.local_file_system import read_file as read_file_tool
    return read_file_tool.execute(absolute_path=absolute_path)

def search_file_content(pattern, path):
    from core.tools.local_file_system import search_file_content as search_file_content_tool
    return search_file_content_tool.execute(pattern=pattern, path=path)

def replace(file_path, old_string, new_string):
    from core.tools.local_file_system import replace as replace_tool
    return replace_tool.execute(file_path=file_path, old_string=old_string, new_string=new_string)

def run_shell_command(command):
    from core.tools.local_file_system import run_shell_command as run_shell_command_tool
    return run_shell_command_tool.execute(command=command)


TEST_FILE_PATH = "/media/seikarii/Nvme/gemini-cli/test_file.txt"
TEST_CONTENT = "Hello Gemini CLI!"
SEARCH_PATTERN = "Gemini"
REPLACE_CONTENT = "Hello World!"

def run_tests():
    """Ejecuta una serie de pruebas para las funciones básicas de la API."""
    print("--- Running write_file test ---")
    write_result = write_file(TEST_FILE_PATH, TEST_CONTENT)
    print(write_result)

    print("\n--- Running read_file test ---")
    read_result = read_file(TEST_FILE_PATH)
    print(read_result)
    # Verificación
    if read_result and read_result.get("content") == TEST_CONTENT:
        print("OK: El contenido leído coincide con el contenido escrito.")
    else:
        print("ERROR: El contenido leído NO coincide con el contenido escrito.")

    print("\n--- Running search_file_content test ---")
    search_result = search_file_content(SEARCH_PATTERN, "/media/seikarii/Nvme/gemini-cli")
    print(search_result)
    # Verificación
    found = False
    if search_result and search_result.get("matches"):
        for match in search_result["matches"]:
            if match.get("file_path") == TEST_FILE_PATH:
                found = True
                break
    if found:
        print(f"OK: search_file_content encontró el patrón '{SEARCH_PATTERN}' en el archivo de prueba.")
    else:
        print(f"ERROR: search_file_content NO encontró el patrón '{SEARCH_PATTERN}' en el archivo de prueba.")

    print("\n--- Running replace test ---")
    replace_result = replace(TEST_FILE_PATH, TEST_CONTENT, REPLACE_CONTENT)
    print(replace_result)

    print("\n--- Running read_file test after replace ---")
    read_after_replace_result = read_file(TEST_FILE_PATH)
    print(read_after_replace_result)
    # Verificación
    if read_after_replace_result and read_after_replace_result.get("content") == REPLACE_CONTENT:
        print("OK: La función de reemplazo parece haber funcionado correctamente.")
    else:
        print("ERROR: La función de reemplazo NO funcionó como se esperaba.")

    print("\n--- Cleaning up test file ---")
    cleanup_result = run_shell_command(f"rm {TEST_FILE_PATH}")
    print(cleanup_result)
    print("\n--- Tests finished ---")

if __name__ == "__main__":
    run_tests()
