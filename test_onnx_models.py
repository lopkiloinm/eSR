#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify ONNX models are valid and loadable
"""
import onnxruntime as ort
import numpy as np
from pathlib import Path


def test_onnx_model(model_path):
    """Test if an ONNX model can be loaded and run"""
    print(f"\n🧪 Testing: {model_path}")
    
    try:
        # Load the model
        session = ort.InferenceSession(str(model_path))
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"  ✅ Model loaded successfully")
        print(f"  📊 Input shape: {input_info.shape}")
        print(f"  📊 Output shape: {output_info.shape}")
        print(f"  🔧 Input name: {input_info.name}")
        print(f"  🔧 Output name: {output_info.name}")
        
        # Test with dummy input
        dummy_input = np.random.rand(1, 1, 64, 64).astype(np.float32)
        input_dict = {input_info.name: dummy_input}
        
        # Run inference
        outputs = session.run(None, input_dict)
        output_shape = outputs[0].shape
        
        print(f"  ✅ Inference successful")
        print(f"  📊 Actual output shape: {output_shape}")
        
        # Verify output values are reasonable
        output_min = np.min(outputs[0])
        output_max = np.max(outputs[0])
        output_mean = np.mean(outputs[0])
        
        print(f"  📈 Output range: [{output_min:.3f}, {output_max:.3f}]")
        print(f"  📈 Output mean: {output_mean:.3f}")
        
        # File size
        file_size = model_path.stat().st_size
        print(f"  💾 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("🔍 Testing ONNX Models")
    print("=" * 50)
    
    onnx_dir = Path("onnx_models")
    
    if not onnx_dir.exists():
        print(f"❌ ONNX models directory not found: {onnx_dir}")
        return
    
    onnx_files = list(onnx_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"❌ No ONNX files found in {onnx_dir}")
        return
    
    print(f"Found {len(onnx_files)} ONNX model(s)")
    
    success_count = 0
    total_count = len(onnx_files)
    
    for onnx_file in sorted(onnx_files):
        if test_onnx_model(onnx_file):
            success_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Successful: {success_count}/{total_count}")
    print(f"  ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"\n🎉 All models are valid and ready for web deployment!")
    else:
        print(f"\n⚠️  Some models have issues. Check the errors above.")


if __name__ == "__main__":
    main() 