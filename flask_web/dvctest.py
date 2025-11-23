# diagnostic.py - Run this to see what's happening
import os
import subprocess
from pathlib import Path

print("=== DIAGNOSTIC SCRIPT ===")

# Get paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

print(f"1. Script location: {SCRIPT_DIR}")
print(f"2. Project root: {PROJECT_ROOT}")
print(f"3. Models directory: {MODELS_DIR}")
print(f"4. Current working directory: {os.getcwd()}")

# Check if directories exist
print(f"\n5. Directory check:")
print(f"   Script dir exists: {SCRIPT_DIR.exists()}")
print(f"   Project root exists: {PROJECT_ROOT.exists()}")
print(f"   Models dir exists: {MODELS_DIR.exists()}")

# List contents of project root
print(f"\n6. Contents of project root:")
try:
    for item in PROJECT_ROOT.iterdir():
        print(f"   - {item.name} (dir: {item.is_dir()})")
except Exception as e:
    print(f"   Error listing project root: {e}")

# Check for .git and .dvc
print(f"\n7. Git and DVC check:")
print(f"   .git exists: {(PROJECT_ROOT / '.git').exists()}")
print(f"   .dvc exists: {(PROJECT_ROOT / '.dvc').exists()}")

# Check for model files directly
print(f"\n8. Model files check:")
model_files = ["YOLOv8m_best.pt", "YOLOv8n.pt"]
for model in model_files:
    model_path = MODELS_DIR / model
    print(f"   {model}: {model_path.exists()}")

# Test DVC manually
print(f"\n9. Testing DVC commands:")
try:
    # Change to project root
    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    print(f"   Changed to: {os.getcwd()}")
    
    # Try DVC status
    result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
    print(f"   DVC status: {result.returncode}")
    if result.stdout:
        print(f"   DVC status output: {result.stdout}")
    
    # Try DVC pull on one file
    result = subprocess.run(["dvc", "pull", "models/YOLOv8m_best.pt.dvc"], capture_output=True, text=True)
    print(f"   DVC pull return code: {result.returncode}")
    if result.stderr:
        print(f"   DVC pull error: {result.stderr}")
    
    os.chdir(original_dir)
    
except Exception as e:
    print(f"   Error testing DVC: {e}")
    os.chdir(original_dir)

print("\n=== END DIAGNOSTIC ===")