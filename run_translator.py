import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.translator_generic import main

if __name__ == "__main__":
    main() 