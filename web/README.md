# edge-SR Web Interface

A web-based implementation of edge-SR super resolution models using ONNX Runtime Web. This allows you to run super resolution inference directly in the browser without any server-side processing.

## Features

- **Browser-based**: No server required, runs entirely in the browser
- **Real-time**: Fast inference using WebAssembly
- **Multiple Models**: Support for various edge-SR architectures
- **Drag & Drop**: Easy image upload interface
- **Performance Monitoring**: Real-time processing statistics
- **Download Results**: Save enhanced images directly

## Setup Instructions

### 1. Convert PyTorch Models to ONNX

First, convert the PyTorch models to ONNX format:

```bash
# Install required packages
pip install torch torchvision onnx

# Convert all models (this may take some time)
python convert_to_onnx.py

# Or convert specific models
python convert_to_onnx.py --model model-files/eSR-TR_s2_K3_C4.model --limit 5
```

### 2. Serve the Web Interface

You can serve the web interface using any HTTP server. Here are a few options:

#### Option A: Python HTTP Server
```bash
cd web
python -m http.server 8000
```

#### Option B: Node.js HTTP Server
```bash
cd web
npx serve .
```

#### Option C: Using our included server
```bash
python serve_web.py
```

### 3. Access the Application

Open your browser and navigate to:
- `http://localhost:8000` (if using Python server)
- `http://localhost:3000` (if using Node.js serve)
- The URL shown by the server you're using

## Usage Guide

### 1. Load a Model
- Select a model from the dropdown menu
- Click "Load Model" and wait for it to download and initialize
- Different models offer different trade-offs between speed and quality

### 2. Upload an Image
- Click "Upload Image" or drag and drop an image file
- Supported formats: PNG, JPG, JPEG, BMP
- Images will be automatically converted to grayscale

### 3. Process the Image
- Click "Enhance Image" to start super resolution
- Processing time depends on image size and model complexity
- Results will appear in the "Enhanced Image" panel

### 4. Download Results
- Click "Download Enhanced Image" to save the result
- Files are saved with model name and timestamp

## Model Information

### Available Architectures

- **eSR-TR**: Transformer-based models with attention mechanisms
- **eSR-MAX**: Max-pooling based models (fastest)
- **eSR-TM**: Temperature-based soft attention models
- **eSR-CNN**: Traditional CNN-based models
- **Bicubic**: Classical interpolation baseline

### Model Naming Convention

Models follow the pattern: `{Architecture}_{Scale}_{Parameters}`

- `s2`, `s3`, `s4`: Upscaling factor (2x, 3x, 4x)
- `K3`, `K5`, `K7`: Kernel size
- `C4`, `C8`, `C16`: Number of channels

### Performance Tips

1. **Start with smaller models** (e.g., `eSR-TR_s2_K3_C4`) for faster processing
2. **Use appropriate scale factors** based on your input image size
3. **Clear browser cache** if you encounter loading issues
4. **Use modern browsers** (Chrome, Firefox, Safari, Edge) for best performance

## Browser Compatibility

- **Chrome/Chromium**: Full support with WebAssembly SIMD
- **Firefox**: Full support
- **Safari**: Full support (macOS/iOS)
- **Edge**: Full support
- **Mobile browsers**: Limited by device performance

## Troubleshooting

### Model Loading Issues
- Ensure ONNX files are properly converted and accessible
- Check browser console for detailed error messages
- Verify network connection for CDN resources

### Performance Issues
- Try smaller models or reduce image size
- Close other browser tabs to free memory
- Use desktop browsers for better performance

### Memory Issues
- Refresh the page to clear GPU/CPU memory
- Process smaller images or use lower-parameter models
- Monitor browser memory usage in developer tools

## Technical Details

### ONNX Runtime Web Configuration
- **Execution Provider**: WebAssembly (WASM)
- **Graph Optimization**: Enabled
- **Memory Management**: Automatic cleanup
- **Threading**: Single-threaded for compatibility

### Image Processing Pipeline
1. **Input**: RGBA image from canvas
2. **Preprocessing**: Convert to grayscale, normalize to [0,1]
3. **Inference**: ONNX model execution
4. **Postprocessing**: Denormalize, convert back to RGBA
5. **Output**: Enhanced image on canvas

## Development

### File Structure
```
web/
├── index.html          # Main HTML interface
├── style.css          # Styling and layout
├── script.js          # JavaScript application logic
├── serve_web.py       # Development server
└── README.md          # This file
```

### Adding New Models
1. Convert PyTorch model to ONNX format
2. Place `.onnx` file in the `onnx_models/` directory
3. Add model option to the HTML select element
4. Update model parsing logic if needed

### Customization
- Modify `style.css` for custom styling
- Update `script.js` for additional features
- Extend model support by updating the conversion script

## License

This web interface follows the same license as the original edge-SR project.

## Citation

If you use this web interface in your research, please cite the original paper:

```bibtex
@inproceedings{eSR,
    title     = {edge--{SR}: Super--Resolution For The Masses},
    author    = {Navarrete~Michelini, Pablo and Lu, Yunhua and Jiang, Xingqun},
    booktitle = {Proceedings of the {IEEE/CVF} Winter Conference on Applications of Computer Vision ({WACV})},
    month     = {January},
    year      = {2022},
    pages     = {1078--1087},
    url       = {https://arxiv.org/abs/2108.10335}
}
``` 