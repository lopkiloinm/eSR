#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for ONNX conversion of edge-SR models
Demonstrates the conversion process and validates the results
"""
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
from pathlib import Path

from convert_to_onnx import convert_model_to_onnx
from models import edgeSR_TR


def create_test_image(size=(64, 64)):
    """Create a simple test image for validation"""
    # Create a simple pattern
    img = np.zeros(size, dtype=np.uint8)
    
    # Add some patterns
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            if (i//8 + j//8) % 2 == 0:
                img[i:i+8, j:j+8] = 255
    
    return img


def test_pytorch_model(model_file, test_image):
    """Test the original PyTorch model"""
    print(f"\n=== Testing PyTorch Model: {model_file} ===")
    
    model_id = model_file.split('.')[-2].split('/')[-1]
    
    # Load PyTorch model
    model = edgeSR_TR(model_id)
    model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=True)
    model.eval()
    
    # Prepare input
    input_tensor = torch.FloatTensor(test_image).unsqueeze(0).unsqueeze(0) / 255.0
    print(f"PyTorch input shape: {input_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        pytorch_output = model(input_tensor)
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
    
    return pytorch_output.numpy()


def test_onnx_model(onnx_file, test_image):
    """Test the converted ONNX model"""
    print(f"\n=== Testing ONNX Model: {onnx_file} ===")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_file)
    
    # Prepare input
    input_tensor = test_image.astype(np.float32)[np.newaxis, np.newaxis, :, :] / 255.0
    print(f"ONNX input shape: {input_tensor.shape}")
    
    # Run inference
    inputs = {session.get_inputs()[0].name: input_tensor}
    onnx_output = session.run(None, inputs)[0]
    
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
    
    return onnx_output


def compare_outputs(pytorch_output, onnx_output, tolerance=1e-5):
    """Compare PyTorch and ONNX outputs"""
    print(f"\n=== Comparing Outputs ===")
    
    # Calculate differences
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Tolerance: {tolerance}")
    
    if max_diff < tolerance:
        print("✅ PASS: Outputs match within tolerance")
        return True
    else:
        print("❌ FAIL: Outputs differ beyond tolerance")
        return False


def save_comparison_images(pytorch_output, onnx_output, prefix="test"):
    """Save comparison images"""
    print(f"\n=== Saving Comparison Images ===")
    
    # Convert to images
    pytorch_img = np.clip(pytorch_output[0, 0] * 255, 0, 255).astype(np.uint8)
    onnx_img = np.clip(onnx_output[0, 0] * 255, 0, 255).astype(np.uint8)
    
    # Save images
    os.makedirs("test_outputs", exist_ok=True)
    
    Image.fromarray(pytorch_img).save(f"test_outputs/{prefix}_pytorch.png")
    Image.fromarray(onnx_img).save(f"test_outputs/{prefix}_onnx.png")
    
    # Save difference image
    diff_img = np.abs(pytorch_img.astype(np.int16) - onnx_img.astype(np.int16))
    diff_img = (diff_img * 10).clip(0, 255).astype(np.uint8)  # Amplify differences
    Image.fromarray(diff_img).save(f"test_outputs/{prefix}_diff.png")
    
    print(f"Saved comparison images to test_outputs/")


def main():
    print("🧪 edge-SR ONNX Conversion Test Suite")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image((64, 64))
    print(f"Created test image: {test_image.shape}")
    
    # Save test image
    os.makedirs("test_outputs", exist_ok=True)
    Image.fromarray(test_image).save("test_outputs/input.png")
    
    # Find a small model to test
    model_files = list(Path("model-files").glob("*s2_K3_C4.model"))
    
    if not model_files:
        print("❌ No suitable test models found. Please ensure model files exist.")
        print("   Looking for models matching pattern: *s2_K3_C4.model")
        return False
    
    # Test the first suitable model
    model_file = str(model_files[0])
    model_name = model_file.split('.')[-2].split('/')[-1]
    
    print(f"\n🔍 Testing model: {model_name}")
    
    try:
        # Test PyTorch model
        pytorch_output = test_pytorch_model(model_file, test_image)
        
        # Convert to ONNX
        print(f"\n🔄 Converting to ONNX...")
        onnx_file = convert_model_to_onnx(model_file, input_size=(1, 1, 64, 64))
        
        # Test ONNX model
        onnx_output = test_onnx_model(onnx_file, test_image)
        
        # Compare outputs
        success = compare_outputs(pytorch_output, onnx_output)
        
        # Save comparison images
        save_comparison_images(pytorch_output, onnx_output, model_name)
        
        if success:
            print(f"\n🎉 SUCCESS: Model {model_name} converted successfully!")
            print(f"   ONNX model saved to: {onnx_file}")
            print(f"   Comparison images saved to: test_outputs/")
        else:
            print(f"\n⚠️  WARNING: Model {model_name} conversion may have issues")
            print(f"   Please check the comparison images in test_outputs/")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 