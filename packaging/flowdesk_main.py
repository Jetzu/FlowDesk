"""PyInstaller entry point (absolute import; ``flowdesk/__main__.py`` uses relative ones)."""
import multiprocessing
import sys

from flowdesk.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
