import os
import sys
import threading
from django.core.management import execute_from_command_line

def start_django():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    # --noreload é essencial no PyInstaller para evitar que crie múltiplos processos
    execute_from_command_line(['manage.py', 'runserver', '127.0.0.1:8000', '--noreload'])

if __name__ == '__main__':
    # Quando integrarmos o pywebview, ele vai rodar na thread principal e o Django nesta thread.
    # Por enquanto, rodamos o Django diretamente.
    print("Iniciando servidor local do Limpador PRO...")
    start_django()
