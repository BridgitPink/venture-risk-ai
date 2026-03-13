# src/run_pipeline.py

import subprocess
import sys


def run_step(script_path: str) -> None:
    print(f"\nRunning {script_path} ...")
    result = subprocess.run([sys.executable, script_path], check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {script_path}")


def main():
    run_step("src/prepare_real_data.py")
    run_step("src/train_embedding_logreg.py")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()