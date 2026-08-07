# LCA_vmtk_pipeline/generate_lca_vmtk_dataset.py

import sys
from pathlib import Path


def main():
    # Resolve paths relative to this script
    base_dir = Path(__file__).resolve().parent.parent
    vmtk_dataset_dir = base_dir / "LCA_branch_control_points" / "vmtk_generated"
    vmtk_output_dir = base_dir / "outputs" / "vmtk_dataset_lca"
    
    # 1. Validation check before calling original script
    ctrl_points_file = vmtk_dataset_dir / "LCA_tree_ctrl_points.npy"
    if not ctrl_points_file.exists():
        print(f"\n[ERROR] VMTK control points file not found: {ctrl_points_file}")
        print("Please run the preprocessor first:\n")
        print("    python -m lca_vessel_tree_generator.LCA_vmtk_pipeline.vmtk_preprocessor")
        sys.exit(1)
        
    print("Launching downstream dataset generator on VMTK control points...")
    print(f"Input Dataset Directory: {vmtk_dataset_dir}")
    print(f"Output Directory:        {vmtk_output_dir}")
    
    # 2. Inject command-line arguments to redirect input/output paths
    # Preserve any other arguments passed by the user (like --lmca-points, etc.)
    new_args = [sys.argv[0]]
    
    # Check if user already provided --dataset-dir or --output in CLI
    has_dataset_dir = any(arg.startswith("--dataset-dir") for arg in sys.argv)
    has_output = any(arg.startswith("--output") for arg in sys.argv)
    
    if not has_dataset_dir:
        new_args.extend(["--dataset-dir", str(vmtk_dataset_dir)])
    if not has_output:
        new_args.extend(["--output", str(vmtk_output_dir)])
        
    new_args.extend(sys.argv[1:])
    sys.argv = new_args
    
    # 3. Import and execute the original generator's main function
    try:
        from lca_vessel_tree_generator.LCA_topology_generator.generate_lca_dataset import main as original_main
        original_main()
    except Exception as e:
        print(f"\n[ERROR] Downstream execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
