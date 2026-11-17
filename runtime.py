"""Project-local SUMO runtime selection; import before sumo_rl or libsumo."""
import os
from pathlib import Path
import sumo

ROOT = Path(__file__).resolve().parent
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = str(Path(sumo.SUMO_HOME) / "bin") + os.pathsep + os.environ["PATH"]
if os.environ.get("CROSSROADS_BACKEND", "libsumo") == "libsumo":
    os.environ["LIBSUMO_AS_TRACI"] = "1"
else:
    os.environ.pop("LIBSUMO_AS_TRACI", None)
import traci  # Establish adapter before any direct libsumo import.
