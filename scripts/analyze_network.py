#!/usr/bin/env python3
"""
Command-line interface for network analysis.

This script provides a convenient way to analyze trained neural networks
from DRL-Finance experiments, including architecture analysis, sensitivity
analysis, diagnostic plots, and specialized convolutional layer analysis.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.network_visualizer import NetworkVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neural network architecture and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with all components
  python scripts/analyze_network.py experiments/my_experiment --analysis-type full
  
  # Basic structure analysis only
  python scripts/analyze_network.py experiments/my_experiment --analysis-type structure
  
  # Analyze Conv1D operations (price processors)
  python scripts/analyze_network.py experiments/my_experiment --analysis-type conv1d
  
  # Analyze Conv2D operations (OHLCV processors)
  python scripts/analyze_network.py experiments/my_experiment --analysis-type conv2d
  
  # Export to ONNX for topology visualization
  python scripts/analyze_network.py experiments/my_experiment --analysis-type onnx
        """
    )
    
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory"
    )
    
    parser.add_argument(
        "--analysis-type",
        choices=["full", "structure", "conv1d", "conv2d", "onnx"],
        default="full",
        help="Type of analysis to perform (default: full)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for analysis results (default: experiment_dir/analysis)"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for sensitivity analysis (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize network visualizer
        visualizer = NetworkVisualizer(args.experiment_dir, args.output_dir)
        
        # Load experiment
        experiment_data = visualizer.load_experiment()
        
        # Run analysis
        results = visualizer.run_analysis(
            analysis_type=args.analysis_type,
            n_samples=args.n_samples
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"NETWORK ANALYSIS COMPLETE: {args.analysis_type.upper()}")
        print(f"{'='*60}")
        
        if args.analysis_type == "full":
            print(f"✓ Architecture Analysis: {len(results.get('structure', {}).get('modules', {}))} modules")
            print(f"✓ Sensitivity Analysis: Complete")
            print(f"✓ Diagnostic Analysis: Complete")
            if "conv1d_analysis" in results:
                conv1d_layers = len([k for k in results["conv1d_analysis"].keys() if k != "filter_responses"])
                print(f"✓ Conv1D Analysis: {conv1d_layers} layers")
            if "conv2d_analysis" in results:
                conv2d_layers = len([k for k in results["conv2d_analysis"].keys() if k != "filter_responses"])
                print(f"✓ Conv2D Analysis: {conv2d_layers} layers")
            if "onnx_analysis" in results:
                onnx_success = results["onnx_analysis"].get("onnx_export", {}).get("success", False)
                print(f"✓ ONNX Export: {'Success' if onnx_success else 'Failed'}")
                
        elif args.analysis_type == "structure":
            print(f"✓ Network Structure: {len(results.get('structure', {}).get('modules', {}))} modules")
            print(f"✓ Parameters: {results.get('structure', {}).get('total_parameters', 0):,}")
            
        elif args.analysis_type == "conv1d":
            if results:
                conv1d_layers = len([k for k in results.keys() if k != "filter_responses"])
                print(f"✓ Conv1D Analysis: {conv1d_layers} layers")
            else:
                print("⚠ No Conv1D layers found")
                
        elif args.analysis_type == "conv2d":
            if results:
                conv2d_layers = len([k for k in results.keys() if k != "filter_responses"])
                print(f"✓ Conv2D Analysis: {conv2d_layers} layers")
            else:
                print("⚠ No Conv2D layers found")
                
        elif args.analysis_type == "onnx":
            onnx_success = results.get("onnx_export", {}).get("success", False)
            if onnx_success:
                onnx_path = results.get("onnx_export", {}).get("onnx_path", "")
                print(f"✓ ONNX Export: Success")
                print(f"✓ File: {onnx_path}")
            else:
                error = results.get("onnx_export", {}).get("error", "Unknown error")
                print(f"✗ ONNX Export Failed: {error}")
        
        print(f"\nOutput directory: {visualizer.output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main() 