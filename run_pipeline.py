import subprocess
import sys
import time


def run_step(script_name, step_description):
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING STEP: {step_description}")
    print(f"   Running {script_name}...")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Run the script and wait for it to finish
    result = subprocess.run([sys.executable, script_name], capture_output=False)

    end_time = time.time()
    duration = end_time - start_time

    if result.returncode == 0:
        print(f"\n✅ SUCCESS: {script_name} finished in {duration:.2f} seconds.")
    else:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}.")
        sys.exit(1)  # Stop the entire pipeline if one step fails


if __name__ == "__main__":
    total_start = time.time()

    print("🔥 INITIATING FRAUD DETECTION PIPELINE 🔥")

    # 1. Generate Synthetic Data
    run_step("syndata.py", "Generating Realistic Synthetic Fraud Data")

    # 2. Load Data into Neo4j
    run_step("kg_loader.py", "Loading Data into Knowledge Graph (Neo4j)")

    # 3. Generate Graph Features (WCC & GraphSAGE)
    run_step("graph_feature_gen.py", "Training GraphSAGE & Extracting Features")

    # 4. Train & Compare Models
    run_step("train_comparison.py", "Training XGBoost Models & Comparing Results")

    total_duration = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_duration:.2f} seconds!")
    print(f"{'=' * 60}")