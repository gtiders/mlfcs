"""Run the project phono3py supercell adapter."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from phono3py_thermal_conductivity import main  # noqa: E402

if __name__ == "__main__":
    main()
