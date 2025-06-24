#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a mega ONNX model that combines multiple edge-SR models
WARNING: This is for demonstration only - not recommended for production!
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse

from models import edgeSR_MAX, edgeSR_TM, edgeSR_CNN, edgeSR_TR, FSRCNN, ESPCN, Classic


class MegaEdgeSRModel(nn.Module):
    """
    Combines multiple edge-SR models into a single model with conditional execution
    WARNING: This is inefficient and not recommended!
    """
    
    def __init__(self, model_configs):
        super().__init__()
        self.models = nn.ModuleDict()
        self.model_configs = model_configs
        
        # Load all models
        for model_id, model_file in model_configs.items():
            print(f"Loading {model_id}...")
            model = self._create_model(model_id)
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            self.models[model_id] = model
    
    def _create_model(self, model_id):
        """Create model instance based on model ID"""
        if model_id.startswith('eSR-MAX_'):
            return edgeSR_MAX(model_id)
        elif model_id.startswith('eSR-TM_'):
            return edgeSR_TM(model_id)
        elif model_id.startswith('eSR-TR_'):
            return edgeSR_TR(model_id)
        elif model_id.startswith('eSR-CNN_'):
            return edgeSR_CNN(model_id)
        elif model_id.startswith('FSRCNN_'):
            return FSRCNN(model_id)
        elif model_id.startswith('ESPCN_'):
            return ESPCN(model_id)
        elif model_id.startswith('Bicubic_'):
            return Classic(model_id)
        else:
            raise ValueError(f"Unknown model type: {model_id}")
    
    def forward(self, x, model_selector):
        """
        Forward pass with model selection
        Args:
            x: Input tensor
            model_selector: Integer indicating which model to use (0, 1, 2, ...)
        """
        model_ids = list(self.models.keys())
        
        # This is where it gets ugly - we need to use conditional execution
        # ONNX doesn't handle this elegantly
        outputs = []
        
        for i, model_id in enumerate(model_ids):
            # Create a mask for this model
            mask = (model_selector == i).float()
            
            # Run the model (always runs, but output is masked)
            model_output = self.models[model_id](x)
            
            # Mask the output
            masked_output = model_output * mask.view(-1, 1, 1, 1)
            outputs.append(masked_output)
        
        # Sum all outputs (only one will be non-zero due to masking)
        return sum(outputs)


def create_mega_model():
    """Create a mega model combining several edge-SR models"""
    
    # Select a few representative models (don't do all 100+!)
    model_configs = {
        'eSR-TR_s2_K3_C4': 'model-files/eSR-TR_s2_K3_C4.model',
        'eSR-MAX_s2_K3_C4': 'model-files/eSR-MAX_s2_K3_C4.model', 
        'eSR-TM_s2_K3_C4': 'model-files/eSR-TM_s2_K3_C4.model',
        'Bicubic_s2': 'model-files/Bicubic_s2.model',
    }
    
    # Check if model files exist
    existing_configs = {}
    for model_id, model_file in model_configs.items():
        if Path(model_file).exists():
            existing_configs[model_id] = model_file
        else:
            print(f"Warning: {model_file} not found, skipping...")
    
    if not existing_configs:
        print("No model files found! Please check your model-files directory.")
        return None
    
    print(f"Creating mega model with {len(existing_configs)} sub-models:")
    for model_id in existing_configs:
        print(f"  - {model_id}")
    
    # Create the mega model
    mega_model = MegaEdgeSRModel(existing_configs)
    
    return mega_model, list(existing_configs.keys())


def export_mega_model_to_onnx():
    """Export the mega model to ONNX format"""
    print("\n" + "="*60)
    print("Creating Mega Edge-SR Model (NOT RECOMMENDED!)")
    print("="*60)
    
    mega_model, model_ids = create_mega_model()
    if mega_model is None:
        return False
    
    # Create dummy inputs
    batch_size = 1
    channels = 1
    height, width = 64, 64
    
    dummy_image = torch.randn(batch_size, channels, height, width)
    dummy_selector = torch.tensor([0], dtype=torch.long)  # Select first model
    
    # Export to ONNX
    onnx_path = "onnx_models/mega_edgesr_model.onnx"
    
    print(f"\nExporting to ONNX: {onnx_path}")
    print("This may take a while and create a large file...")
    
    try:
        torch.onnx.export(
            mega_model,
            (dummy_image, dummy_selector),
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['image', 'model_selector'],
            output_names=['enhanced_image'],
            dynamic_axes={
                'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'enhanced_image': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        # Check file size
        file_size = Path(onnx_path).stat().st_size
        print(f"\n✅ Mega model exported successfully!")
        print(f"   File: {onnx_path}")
        print(f"   Size: {file_size / 1024:.1f} KB")
        print(f"   Models included: {len(model_ids)}")
        
        print(f"\n📋 Model selector values:")
        for i, model_id in enumerate(model_ids):
            print(f"   {i}: {model_id}")
        
        print(f"\n⚠️  WARNING: This approach has major drawbacks:")
        print(f"   - Large file size (all models combined)")
        print(f"   - All models load into memory simultaneously") 
        print(f"   - Complex conditional execution")
        print(f"   - Poor browser performance")
        print(f"   - Difficult to maintain")
        
        print(f"\n💡 RECOMMENDATION: Use the dynamic loading approach instead!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create a mega ONNX model (NOT RECOMMENDED!)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This creates an inefficient mega-model for demonstration only.

The dynamic loading approach in the web interface is much better:
- Smaller file sizes
- Faster loading
- Better memory usage  
- Easier maintenance

Only use this if you have very specific requirements!
        """
    )
    
    args = parser.parse_args()
    
    success = export_mega_model_to_onnx()
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"   1. Test the mega model with ONNX Runtime")
        print(f"   2. Update web interface to use model selector")
        print(f"   3. Compare performance vs. dynamic loading")
        print(f"   4. Realize dynamic loading is better! 😉")
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main() 