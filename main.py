import os
import sys
import threading
import webbrowser
import tkinter as tk
from tkinter import messagebox

def start_django():
    from django.core.management import execute_from_command_line
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    execute_from_command_line(['manage.py', 'runserver', '127.0.0.1:8000', '--noreload'])

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor e Removedor de Templates")
        self.root.geometry("350x280")
        
        self.server_thread = None
        
        tk.Label(root, text="Editor e Removedor\nde Templates", font=("Helvetica", 14, "bold")).pack(pady=15)
        
        self.btn_iniciar = tk.Button(root, text="INICIAR", command=self.iniciar_servidor, width=25, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_iniciar.pack(pady=10)
        
        self.btn_templates = tk.Button(root, text="ABRIR PASTA DE TEMPLATES", command=self.abrir_templates, width=25, font=("Helvetica", 10))
        self.btn_templates.pack(pady=10)
        
        self.btn_fechar = tk.Button(root, text="FECHAR", command=self.fechar, width=25, bg="#f44336", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_fechar.pack(pady=10)

    def iniciar_servidor(self):
        if self.server_thread is None or not self.server_thread.is_alive():
            self.server_thread = threading.Thread(target=start_django, daemon=True)
            self.server_thread.start()
            self.btn_iniciar.config(state=tk.DISABLED, text="SERVIDOR RODANDO", bg="#81C784")
            self.root.after(2000, lambda: webbrowser.open("http://127.0.0.1:8000/"))
        else:
            messagebox.showinfo("Info", "Servidor já está rodando!")

    def abrir_templates(self):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
        from django.conf import settings
        
        templates_dir = os.path.join(settings.MEDIA_ROOT, 'templates')
        
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
            
        if os.name == 'nt':
            os.startfile(templates_dir)
        else:
            messagebox.showinfo("Pasta de Templates", f"A pasta de templates está em:\n{templates_dir}")

    def fechar(self):
        self.root.quit()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
