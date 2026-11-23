"""Inicia localmente o servidor de processamento (porta 5000) e o servidor web do client (porta 8080) sem Docker."""
import os
import subprocess
import sys
import time
import webbrowser

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT_DIR
    env.setdefault("SERVER_URL", "http://127.0.0.1:5000")

    server_proc = subprocess.Popen([sys.executable, os.path.join(ROOT_DIR, "server", "app.py")], env=env)
    time.sleep(1.2)
    client_proc = subprocess.Popen([sys.executable, os.path.join(ROOT_DIR, "client", "app.py")], env=env)

    url = "http://127.0.0.1:8080"
    print(f"\nSistema iniciado! Acesse no navegador: {url}")
    print("Pressione Ctrl+C para encerrar ambos os servidores.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server_proc.wait()
        client_proc.wait()
    except KeyboardInterrupt:
        print("\nEncerrando servidores...")
        client_proc.terminate()
        server_proc.terminate()


if __name__ == "__main__":
    main()
