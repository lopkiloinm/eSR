#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Conversion Script for edge-SR Models
Converts PyTorch models to ONNX format for web deployment
"""
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import argparse
import os

from models import edgeSR_MAX, edgeSR_TM, edgeSR_CNN, edgeSR_TR, FSRCNN, ESPCN, Classic


def convert_model_to_onnx(model_file, output_dir="onnx_models", input_size=(1, 1, 224, 224)):
    """
    Convert a PyTorch model to ONNX format
    
    Args:
        model_file (str): Path to the PyTorch model file
        output_dir (str): Directory to save ONNX models
        input_size (tuple): Input tensor size (batch, channels, height, width)
    """
    model_id = model_file.split('.')[-2].split('/')[-1]
    print(f"\nConverting model: {model_id}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the appropriate model architecture
    # Use CPU for ONNX export (no CUDA dependencies for web deployment)
    device = torch.device('cpu')
    
    with torch.no_grad():
        # Initialize model based on model_id
        if model_id.startswith('eSR-MAX_'):
            model = edgeSR_MAX(model_id)
        elif model_id.startswith('eSR-TM_'):
            model = edgeSR_TM(model_id)
        elif model_id.startswith('eSR-TR_'):
            model = edgeSR_TR(model_id)
        elif model_id.startswith('eSR-CNN_'):
            model = edgeSR_CNN(model_id)
        elif model_id.startswith('FSRCNN_'):
            model = FSRCNN(model_id)
        elif model_id.startswith('ESPCN_'):
            model = ESPCN(model_id)
        elif model_id.startswith('Bicubic_'):
            model = Classic(model_id)
        else:
            raise ValueError(f"Unknown model type: {model_id}")
        
        # Load model weights
        model.load_state_dict(
            torch.load(model_file, map_location='cpu'),
            strict=True
        )
        # Enforce strict float32 for Apple Metal and WebGPU compatibility
        # Apple Metal doesn't support fp64 natively, WebGPU prefers fp32
        model.to(device).float().eval()
        
        # Ensure all parameters are fp32 (important for Apple Metal)
        for param in model.parameters():
            param.data = param.data.float()
        
        # Create dummy input with explicit fp32 dtype (no fp64!)
        dummy_input = torch.randn(input_size, device=device, dtype=torch.float32)
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_id}.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Good compatibility with WebGPU/WebGL/Apple Metal
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            # Optimizations for Apple Metal and WebGPU
            keep_initializers_as_inputs=False,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )
        
        print(f"✓ Exported: {onnx_path}")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"✓ ONNX model verification passed")
        except ImportError:
            print("⚠ ONNX package not found. Install with: pip install onnx")
        except Exception as e:
            print(f"⚠ ONNX verification failed: {e}")
        
        return onnx_path


def convert_all_models(model_dir="model-files", output_dir="onnx_models", limit=None):
    """
    Convert all models in the model directory to ONNX format
    
    Args:
        model_dir (str): Directory containing PyTorch models
        output_dir (str): Directory to save ONNX models
        limit (int): Limit number of models to convert (for testing)
    """
    model_files = list(Path(model_dir).glob("*.model"))
    
    if limit:
        model_files = model_files[:limit]
    
    print(f"Found {len(model_files)} model files")
    
    successful_conversions = []
    failed_conversions = []
    
    for model_file in model_files:
        try:
            onnx_path = convert_model_to_onnx(str(model_file), output_dir)
            successful_conversions.append((str(model_file), onnx_path))
        except Exception as e:
            print(f"✗ Failed to convert {model_file}: {e}")
            failed_conversions.append((str(model_file), str(e)))
    
    print(f"\n=== Conversion Summary ===")
    print(f"Successful: {len(successful_conversions)}")
    print(f"Failed: {len(failed_conversions)}")
    
    if failed_conversions:
        print("\nFailed conversions:")
        for model_file, error in failed_conversions:
            print(f"  {model_file}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Convert edge-SR models to ONNX format")
    parser.add_argument("--model", type=str, help="Specific model file to convert")
    parser.add_argument("--model-dir", type=str, default="model-files", help="Directory containing model files")
    parser.add_argument("--output-dir", type=str, default="onnx_models", help="Output directory for ONNX models")
    parser.add_argument("--limit", type=int, help="Limit number of models to convert")
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 1, 224, 224], 
                        help="Input size as batch channels height width")
    
    args = parser.parse_args()
    
    if args.model:
        # Convert single model
        convert_model_to_onnx(args.model, args.output_dir, tuple(args.input_size))
    else:
        # Convert all models
        convert_all_models(args.model_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main() 