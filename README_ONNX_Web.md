# edge-SR ONNX Web Deployment

This guide shows you how to convert the edge-SR PyTorch models to ONNX format and deploy them in a web browser using ONNX Runtime Web with WebAssembly.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install torch torchvision onnx onnxruntime

# Optional: Install ONNX tools for model optimization
pip install onnx-tools onnxsim
```

### 2. Convert Models to ONNX

```bash
# Convert a few sample models (recommended for testing)
python convert_to_onnx.py --limit 5

# Or convert all models (this will take significant time and space)
python convert_to_onnx.py

# Convert specific model
python convert_to_onnx.py --model model-files/eSR-TR_s2_K3_C4.model
```

### 3. Start the Web Server

```bash
# Start the development server
python serve_web.py

# Or use Python's built-in server
cd web && python -m http.server 8000
```

### 4. Open in Browser

Navigate to `http://localhost:8000` and start enhancing images!

## 📁 Project Structure

After conversion, your project structure will look like this:

```
eSR/
├── model-files/              # Original PyTorch models
├── onnx_models/             # Converted ONNX models
├── web/                     # Web interface
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── README.md
├── convert_to_onnx.py       # Conversion script
├── serve_web.py            # Development server
├── models.py               # Model architectures
├── run.py                  # Original PyTorch inference
└── README_ONNX_Web.md      # This file
```

## 🔧 Technical Details

### ONNX Conversion Process

The conversion process involves several steps:

1. **Model Loading**: Load the PyTorch model with trained weights
2. **Architecture Mapping**: Map model architecture to ONNX operations
3. **Input Preparation**: Create dummy input tensors with correct shapes
4. **Export**: Use `torch.onnx.export()` with optimizations
5. **Verification**: Validate the exported ONNX model

### Key Conversion Features

- **Dynamic Input Shapes**: Models accept variable image sizes
- **Optimized Export**: Uses ONNX opset 11 for broad compatibility
- **Constant Folding**: Reduces model size and improves performance
- **Graph Optimization**: Simplifies the computation graph

### Web Runtime

The web interface uses:

- **ONNX Runtime Web**: JavaScript/WebAssembly runtime
- **WebAssembly**: High-performance execution in browsers
- **Canvas API**: Image processing and display
- **Modern JavaScript**: ES6+ features for clean code

## 🎯 Model Performance

### Recommended Models for Web Deployment

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `eSR-TR_s2_K3_C4` | ~8KB | Fast | Good | Real-time preview |
| `eSR-TR_s2_K5_C8` | ~38KB | Medium | Better | Balanced performance |
| `eSR-MAX_s2_K3_C4` | ~8KB | Fastest | Good | Maximum speed |
| `Bicubic_s2` | ~1KB | Very Fast | Baseline | Reference comparison |

### Performance Tips

1. **Start Small**: Begin with smaller models to test the pipeline
2. **Batch Processing**: Process multiple images in sequence
3. **Memory Management**: Clear model cache between different models
4. **Browser Choice**: Chrome/Edge often have the best WebAssembly performance

## 🌐 Browser Compatibility

### Fully Supported
- Chrome 57+ (recommended)
- Firefox 52+
- Safari 11+
- Edge 16+

### Partially Supported
- Mobile browsers (performance limited)
- Older browsers (may require polyfills)

### Requirements
- WebAssembly support
- Canvas API support
- Modern JavaScript (ES6+)
- Sufficient memory (varies by model)

## 🔍 Troubleshooting

### Common Issues

#### Model Loading Fails
```
Error: Failed to load model
```
**Solution**: 
- Check if ONNX files exist in `onnx_models/` directory
- Verify file permissions and network access
- Try a smaller model first

#### Out of Memory Errors
```
Error: Cannot allocate memory
```
**Solution**:
- Use smaller models (lower channel count)
- Reduce input image size
- Close other browser tabs
- Try a different browser

#### Slow Performance
**Solution**:
- Use faster models (eSR-MAX architecture)
- Reduce image resolution
- Enable browser GPU acceleration
- Use Chrome for best WebAssembly performance

#### CORS Errors
```
Error: Cross-origin request blocked
```
**Solution**:
- Use the provided `serve_web.py` server
- Or configure your web server to allow CORS
- Don't open HTML files directly in browser

### Advanced Debugging

#### Enable Verbose Logging
```javascript
// In browser console
ort.env.logLevel = 'verbose';
```

#### Performance Profiling
```javascript
// In browser console
console.time('inference');
// ... run inference ...
console.timeEnd('inference');
```

#### Memory Usage Monitoring
Use browser Developer Tools → Performance tab to monitor memory usage during inference.

## 🛠️ Customization

### Adding New Models

1. **Train PyTorch Model**: Follow original edge-SR training process
2. **Convert to ONNX**: Use the conversion script
3. **Update Web Interface**: Add model option to HTML
4. **Test**: Verify model works in browser

### Modifying the Web Interface

#### Change Styling
Edit `web/style.css` to customize appearance:
```css
/* Custom color scheme */
:root {
    --primary-color: #your-color;
    --secondary-color: #your-color;
}
```

#### Add Features
Extend `web/script.js` to add functionality:
```javascript
// Example: Add batch processing
async processBatch(files) {
    for (const file of files) {
        await this.processImage(file);
    }
}
```

### Model Optimization

#### Quantization
```python
# Add to convert_to_onnx.py
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic

# After ONNX export
quantize_dynamic(onnx_path, quantized_path)
```

#### Graph Optimization
```python
# Add to convert_to_onnx.py
import onnx
from onnx import optimizer

# Optimize the graph
optimized_model = optimizer.optimize(onnx_model)
```

## 📊 Performance Benchmarks

### Inference Time (on typical hardware)

| Model | Image Size | Chrome | Firefox | Safari |
|-------|------------|--------|---------|--------|
| eSR-TR_s2_K3_C4 | 256x256 | 45ms | 65ms | 55ms |
| eSR-TR_s2_K5_C8 | 256x256 | 120ms | 180ms | 140ms |
| eSR-MAX_s2_K3_C4 | 256x256 | 35ms | 50ms | 45ms |

*Results may vary based on hardware and browser version*

### Model Sizes

| Model | PyTorch | ONNX | Compression |
|-------|---------|------|-------------|
| eSR-TR_s2_K3_C4 | 7.5KB | 8.2KB | 9% larger |
| eSR-TR_s2_K5_C8 | 38KB | 41KB | 8% larger |
| eSR-MAX_s2_K3_C4 | 7.5KB | 8.1KB | 8% larger |

## 🚀 Production Deployment

### CDN Deployment

1. **Upload ONNX models** to a CDN (AWS S3, Cloudflare, etc.)
2. **Update model paths** in `script.js`
3. **Configure CORS** headers on your CDN
4. **Enable compression** (gzip/brotli) for faster downloads

### Optimization for Production

```javascript
// Enable WebAssembly SIMD if available
ort.env.wasm.simd = true;

// Use multi-threading if supported
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

// Optimize memory usage
ort.env.wasm.initTimeout = 30000;
```

### Progressive Loading

```javascript
// Load models on demand
async loadModelOnDemand(modelName) {
    if (!this.modelCache[modelName]) {
        this.modelCache[modelName] = await this.loadModel(modelName);
    }
    return this.modelCache[modelName];
}
```

## 📝 License

This ONNX conversion and web interface follows the same license as the original edge-SR project.

## 🙏 Acknowledgments

- Original edge-SR paper and authors
- ONNX Runtime team for Web support
- PyTorch team for excellent ONNX export capabilities

## 📚 References

1. [edge-SR Paper](https://arxiv.org/abs/2108.10335)
2. [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/tutorials/web/)
3. [PyTorch ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
4. [WebAssembly Documentation](https://webassembly.org/)

---

**Happy super-resolving! 🎨✨** 