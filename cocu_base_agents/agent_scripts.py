import os
from pathlib import Path


def agent_scripts():
    root = Path(os.path.dirname(os.path.realpath(__file__)))
    return {
        "NewAgent": root / "new_agent" / "new_agent.py",
        "MRBTPAgent": root / "new_agent" / "mrbtp_agent.py",
    }
