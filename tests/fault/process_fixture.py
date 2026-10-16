"""Synthetic long-running process tree, launched only in isolated fault tests."""
import json
from pathlib import Path
import os
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid]), encoding="utf-8")
time.sleep(30)
