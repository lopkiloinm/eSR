#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze edge-SR paper results to find the recommended model
Based on the paper's evaluation metrics and trade-offs
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_paper_results():
    """Load the paper's test results from tests.pkl"""
    if not Path('data/tests.pkl').exists():
        print("❌ tests.pkl not found in data/ directory")
        return None
    
    with open('data/tests.pkl', 'rb') as f:
        return pickle.load(f)


def analyze_edge_sr_models(results):
    """Analyze eSR models to find the paper's recommended configuration"""
    
    print("🔍 Analyzing edge-SR Paper Results")
    print("=" * 60)
    
    # Filter for main eSR models (excluding baselines like Bicubic, FSRCNN, ESPCN)
    esr_models = {}
    for model_name, metrics in results.items():
        if model_name.startswith(('eSR-TR_', 'eSR-MAX_', 'eSR-TM_', 'eSR-CNN_')):
            esr_models[model_name] = metrics
    
    print(f"Found {len(esr_models)} eSR models in paper results")
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for model_name, metrics in esr_models.items():
        row = {'model': model_name}
        row.update(metrics)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Focus on scale=2 models (most common use case)
    s2_models = df[df['model'].str.contains('_s2_')]
    
    if len(s2_models) == 0:
        print("No s2 models found, analyzing all models...")
        s2_models = df
    
    print(f"\nAnalyzing {len(s2_models)} scale-2 models:")
    
    # Key metrics for quality vs speed trade-off
    quality_metrics = ['psnr_Set5', 'ssim_Set5', 'lpips_Set5']  # Lower LPIPS is better
    speed_metrics = ['speed_AGX', 'speed_MaxQ', 'speed_RPI']
    efficiency_metrics = ['parameters', 'power_AGX', 'memory_MaxQ']
    
    # Analyze each architecture
    architectures = ['eSR-TR', 'eSR-MAX', 'eSR-TM', 'eSR-CNN']
    
    print("\n📊 Performance Summary by Architecture:")
    print("-" * 80)
    
    best_models = {}
    
    for arch in architectures:
        arch_models = s2_models[s2_models['model'].str.startswith(arch)]
        
        if len(arch_models) == 0:
            continue
            
        print(f"\n{arch} Models ({len(arch_models)} variants):")
        
        # Find best quality model in this architecture
        if 'psnr_Set5' in arch_models.columns:
            best_quality_idx = arch_models['psnr_Set5'].idxmax()
            best_quality = arch_models.loc[best_quality_idx]
            
            # Find fastest model in this architecture  
            if 'speed_AGX' in arch_models.columns:
                best_speed_idx = arch_models['speed_AGX'].idxmin()  # Lower is faster
                best_speed = arch_models.loc[best_speed_idx]
                
                print(f"  Best Quality: {best_quality['model']}")
                print(f"    PSNR: {best_quality.get('psnr_Set5', 'N/A'):.2f}")
                print(f"    Params: {best_quality.get('parameters', 'N/A')}")
                print(f"    Speed: {best_quality.get('speed_AGX', 'N/A'):.1f}ms")
                
                print(f"  Fastest: {best_speed['model']}")
                print(f"    PSNR: {best_speed.get('psnr_Set5', 'N/A'):.2f}")
                print(f"    Params: {best_speed.get('parameters', 'N/A')}")
                print(f"    Speed: {best_speed.get('speed_AGX', 'N/A'):.1f}ms")
                
                # Calculate quality/speed balance
                arch_models_copy = arch_models.copy()
                if 'psnr_Set5' in arch_models_copy.columns and 'speed_AGX' in arch_models_copy.columns:
                    # Normalize metrics (higher PSNR is better, lower speed is better)
                    arch_models_copy['psnr_norm'] = (arch_models_copy['psnr_Set5'] - arch_models_copy['psnr_Set5'].min()) / (arch_models_copy['psnr_Set5'].max() - arch_models_copy['psnr_Set5'].min())
                    arch_models_copy['speed_norm'] = 1 - (arch_models_copy['speed_AGX'] - arch_models_copy['speed_AGX'].min()) / (arch_models_copy['speed_AGX'].max() - arch_models_copy['speed_AGX'].min())
                    
                    # Balance score (equal weight to quality and speed)
                    arch_models_copy['balance_score'] = (arch_models_copy['psnr_norm'] + arch_models_copy['speed_norm']) / 2
                    
                    best_balance_idx = arch_models_copy['balance_score'].idxmax()
                    best_balance = arch_models_copy.loc[best_balance_idx]
                    
                    print(f"  Best Balance: {best_balance['model']}")
                    print(f"    PSNR: {best_balance.get('psnr_Set5', 'N/A'):.2f}")
                    print(f"    Params: {best_balance.get('parameters', 'N/A')}")
                    print(f"    Speed: {best_balance.get('speed_AGX', 'N/A'):.1f}ms")
                    print(f"    Balance Score: {best_balance['balance_score']:.3f}")
                    
                    best_models[arch] = {
                        'model': best_balance['model'],
                        'psnr': best_balance.get('psnr_Set5', 0),
                        'speed': best_balance.get('speed_AGX', float('inf')),
                        'params': best_balance.get('parameters', 0),
                        'balance_score': best_balance['balance_score']
                    }
    
    # Overall recommendation
    print("\n🏆 PAPER'S RECOMMENDED MODELS:")
    print("=" * 60)
    
    if best_models:
        # Sort by balance score
        sorted_models = sorted(best_models.items(), key=lambda x: x[1]['balance_score'], reverse=True)
        
        print("Ranking by Quality/Speed Balance:")
        for i, (arch, model_info) in enumerate(sorted_models, 1):
            print(f"\n{i}. {model_info['model']} ({arch})")
            print(f"   PSNR: {model_info['psnr']:.2f} dB")
            print(f"   Speed: {model_info['speed']:.1f} ms")
            print(f"   Parameters: {model_info['params']:,}")
            print(f"   Balance Score: {model_info['balance_score']:.3f}")
            
            if i == 1:
                print("   ⭐ RECOMMENDED FOR WEB DEPLOYMENT")
        
        return sorted_models[0][1]['model']  # Return top recommendation
    
    return None


def get_recommended_model_info(recommended_model):
    """Get detailed information about the recommended model"""
    if not recommended_model:
        return None
    
    # Parse model name to extract parameters
    parts = recommended_model.split('_')
    
    info = {
        'name': recommended_model,
        'architecture': parts[0],
        'scale': int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith('s') else 2,
        'kernel_size': None,
        'channels': None,
        'file_path': f"model-files/{recommended_model}.model"
    }
    
    # Extract architecture-specific parameters
    for part in parts[2:]:
        if part.startswith('K'):
            info['kernel_size'] = int(part[1:])
        elif part.startswith('C'):
            info['channels'] = int(part[1:])
        elif part.startswith('D'):
            info['depth'] = int(part[1:])
        elif part.startswith('S'):
            info['width'] = int(part[1:])
    
    return info


def main():
    print("📊 Analyzing edge-SR Paper Results")
    print("Finding the optimal model for quality/speed balance")
    print("=" * 60)
    
    # Load results
    results = load_paper_results()
    if results is None:
        return
    
    print(f"Loaded results for {len(results)} models")
    
    # Analyze and find recommended model
    recommended_model = analyze_edge_sr_models(results)
    
    if recommended_model:
        print(f"\n🎯 FINAL RECOMMENDATION: {recommended_model}")
        
        model_info = get_recommended_model_info(recommended_model)
        if model_info:
            print(f"\n📋 Model Details:")
            print(f"   Architecture: {model_info['architecture']}")
            print(f"   Scale Factor: {model_info['scale']}x")
            print(f"   Kernel Size: {model_info.get('kernel_size', 'N/A')}")
            print(f"   Channels: {model_info.get('channels', 'N/A')}")
            print(f"   Model File: {model_info['file_path']}")
            
            # Check if model file exists
            if Path(model_info['file_path']).exists():
                print(f"   ✅ Model file found")
            else:
                print(f"   ❌ Model file not found")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Convert this model: python convert_to_onnx.py --model {model_info['file_path']}")
        print(f"   2. Update web interface to use this as default")
        print(f"   3. Test performance with WebGPU")
    else:
        print("\n❌ Could not determine recommended model from paper results")


if __name__ == "__main__":
    main() 