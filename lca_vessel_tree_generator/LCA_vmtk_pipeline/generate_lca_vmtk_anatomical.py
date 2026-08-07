# lca_vessel_tree_generator/LCA_vmtk_pipeline/generate_lca_vmtk_anatomical.py

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Resolve paths relative to this script
    # This script is at vessel_tree_generator/lca_vessel_tree_generator/LCA_vmtk_pipeline/generate_lca_vmtk_anatomical.py
    # So repo_root is vessel_tree_generator/
    repo_root = Path(__file__).resolve().parent.parent.parent
    
    # Input/output paths relative to repo_root
    vmtk_generated_dir = Path("lca_vessel_tree_generator") / "LCA_branch_control_points" / "vmtk_generated"
    original_metadata_dir = Path("lca_vessel_tree_generator") / "LCA_branch_control_points" / "generated"
    
    output_anatomical = Path("outputs") / "vmtk_dataset_lca_anatomical"
    output_pipeline = Path("outputs") / "vmtk_dataset_lca_anatomical_pipeline"
    output_synthetic = Path("outputs") / "vmtk_dataset_lca_anatomical_synthetic"
    
    print("=" * 80)
    print("RUNNING VMTK ANATOMICAL PIPELINE")
    print(f"Working directory (repo root): {repo_root}")
    print("=" * 80)
    
    # Run Step 1: Reorientation
    print("\n--- STEP 1: Anatomical Reorientation ---")
    cmd_reorient = [
        sys.executable, "-m", "lca_vessel_tree_generator.LCA_topology_generator.anatomical_lca_reorientation",
        "--dataset-dir", str(vmtk_generated_dir),
        "--output", str(output_anatomical),
        "--include-invalid"
    ]
    print(f"Running: {' '.join(cmd_reorient)}")
    res = subprocess.run(cmd_reorient, cwd=repo_root)
    if res.returncode != 0:
        print("[ERROR] STEP 1 (Reorientation) failed.")
        sys.exit(res.returncode)
        
    # Run Step 2: Downstream Pipeline
    print("\n--- STEP 2: Anatomical Downstream Pipeline ---")
    cmd_pipeline = [
        sys.executable, "-m", "lca_vessel_tree_generator.LCA_topology_generator.anatomical_lca_pipeline",
        "--input-dir", str(output_anatomical),
        "--output-dir", str(output_pipeline),
        "--metadata-dir", str(original_metadata_dir)
    ]
    print(f"Running: {' '.join(cmd_pipeline)}")
    res = subprocess.run(cmd_pipeline, cwd=repo_root)
    if res.returncode != 0:
        print("[ERROR] STEP 2 (Pipeline) failed.")
        sys.exit(res.returncode)
        
    # Run Step 3: Controlled Synthetic Generation
    print("\n--- STEP 3: Controlled Synthetic Generation ---")
    cmd_synthetic = [
        sys.executable, "-m", "lca_vessel_tree_generator.LCA_topology_generator.controlled_anatomical_lca_synthetic",
        "--input-dir", str(output_anatomical),
        "--output-dir", str(output_synthetic),
        "--metadata-dir", str(original_metadata_dir)
    ]
    print(f"Running: {' '.join(cmd_synthetic)}")
    res = subprocess.run(cmd_synthetic, cwd=repo_root)
    if res.returncode != 0:
        print("[ERROR] STEP 3 (Synthetic) failed.")
        sys.exit(res.returncode)
        
    print("\n" + "=" * 80)
    print("VMTK ANATOMICAL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Outputs created under outputs/ in:")
    print(f"  - {output_anatomical}")
    print(f"  - {output_pipeline}")
    print(f"  - {output_synthetic}")
    print("=" * 80)

if __name__ == "__main__":
    main()
