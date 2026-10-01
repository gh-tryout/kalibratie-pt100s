"""Start de PT100-kalibratie Streamlit-app, ook als Windows-exe zonder Python."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _port_is_taken(port: int) -> bool:
    """True als localhost of het netwerk die poort al gebruikt."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def find_free_port(host: str = "127.0.0.1", preferred: int = 8525) -> int:
    bind_host = "0.0.0.0" if host == "0.0.0.0" else host
    for port in range(preferred, preferred + 20):
        if not _port_is_taken(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def open_browser(url: str, delay: float = 2.0) -> None:
    def _open() -> None:
        time.sleep(delay)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    if sys.version_info >= (3, 14) and not getattr(sys, "frozen", False):
        print(
            "Python 3.14 crasht met Streamlit/pandas op deze app.\n"
            "Start via start.bat of start_website.bat (Python 3.12 in .venv).\n"
        )
        input("Druk op Enter om af te sluiten...")
        sys.exit(1)

    web_mode = "--web" in sys.argv or os.environ.get("PT100_WEB", "").lower() in {
        "1",
        "true",
        "yes",
    }
    host = os.environ.get("PT100_HOST", "0.0.0.0" if web_mode else "127.0.0.1")

    root = resource_dir()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    os.environ["PT100_DATA_DIR"] = str(exe_dir())
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_LOGGER_LEVEL"] = "info"

    script = root / "app.py"
    if not script.is_file():
        print(f"Kan app.py niet vinden in:\n  {root}")
        input("Druk op Enter om af te sluiten...")
        sys.exit(1)

    port = find_free_port(host=host)
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = host
    os.environ["STREAMLIT_BROWSER_SERVER_PORT"] = str(port)
    # Browser opent altijd localhost; netwerkclients gebruiken het LAN-IP
    os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "127.0.0.1"

    ssl_cert = root / ".ssl" / "cert.pem"
    ssl_key = root / ".ssl" / "key.pem"
    use_ssl = ssl_cert.is_file() and ssl_key.is_file()
    scheme = "https" if use_ssl else "http"

    local_url = f"{scheme}://127.0.0.1:{port}"
    print("PT100-kalibratie vs referentie")
    print(f"Python {sys.version.split()[0]}")
    print(f"Lokaal: {local_url}")
    if web_mode or host == "0.0.0.0":
        print(f"Netwerk: {scheme}://<dit-pc-ip>:{port}")
        print("Andere computers op het netwerk kunnen bestanden uploaden via die URL.")
    print("Laat dit venster open zolang je de app gebruikt.")
    print("Sluit het venster om de app te stoppen.\n")
    open_browser(local_url)

    from streamlit import config as st_config
    from streamlit.web import bootstrap

    script_path = str(script.resolve())
    st_config._main_script_path = script_path

    flag_options = {
        "global_developmentMode": False,
        "server_port": port,
        "server_address": host,
        "server_headless": True,
        "browser_gatherUsageStats": False,
        "browser_serverPort": port,
        "browser_serverAddress": "127.0.0.1",
        "server_fileWatcherType": "none",
        "logger_level": "info",
        "client_toolbarMode": "minimal",
        "server_maxUploadSize": 200,
    }
    if use_ssl:
        flag_options["server_sslCertFile"] = str(ssl_cert)
        flag_options["server_sslKeyFile"] = str(ssl_key)
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(script_path, False, [], flag_options)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFout bij starten: {exc}")
        input("Druk op Enter om af te sluiten...")
        raise
