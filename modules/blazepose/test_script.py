from pathlib import Path
import os
import dotenv
dotenv.load_dotenv()
base_dir = Path(os.getenv("DISCOVER_DATA_DIR"))
out_dir = Path(os.getenv("DISCOVER_TEST_DIR"))

print(base_dir)