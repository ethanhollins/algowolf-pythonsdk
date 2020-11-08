import os
PATH = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(PATH, 'scripts')

from .app.cli import app

