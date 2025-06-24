class EdgeSRProcessor {
    constructor() {
        this.session = null;
        this.modelInfo = null;
        this.isProcessing = false;
        this.modelCache = new Map(); // Add model caching
        
        this.initializeElements();
        this.setupEventListeners();
        this.updateStatus('Ready to load model');
    }
    
    initializeElements() {
        this.elements = {
            modelSelect: document.getElementById('modelSelect'),
            loadModelBtn: document.getElementById('loadModelBtn'),
            imageInput: document.getElementById('imageInput'),
            processBtn: document.getElementById('processBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            colorMode: document.getElementById('colorMode'),
            statusText: document.getElementById('statusText'),
            progressFill: document.getElementById('progressFill'),
            originalCanvas: document.getElementById('originalCanvas'),
            enhancedCanvas: document.getElementById('enhancedCanvas'),
            originalInfo: document.getElementById('originalInfo'),
            enhancedInfo: document.getElementById('enhancedInfo'),
            processingTime: document.getElementById('processingTime'),
            modelSize: document.getElementById('modelSize'),
            upscaleFactor: document.getElementById('upscaleFactor')
        };
    }
    
    setupEventListeners() {
        this.elements.modelSelect.addEventListener('change', () => {
            this.elements.loadModelBtn.disabled = !this.elements.modelSelect.value;
        });
        
        // Enable load button by default since we have a recommended model pre-selected
        this.elements.loadModelBtn.disabled = false;
        
        this.elements.loadModelBtn.addEventListener('click', () => {
            this.loadModel();
        });
        
        this.elements.imageInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.loadImage(e.target.files[0]);
            }
        });
        
        this.elements.processBtn.addEventListener('click', () => {
            this.processImage();
        });
        
        this.elements.downloadBtn.addEventListener('click', () => {
            this.downloadImage();
        });
    }
    
    updateStatus(message, progress = 0) {
        this.elements.statusText.textContent = message;
        this.elements.progressFill.style.width = `${progress}%`;
    }
    
    async loadModel() {
        const modelName = this.elements.modelSelect.value;
        if (!modelName) return;
        
        try {
            // Check if model is already cached
            if (this.modelCache.has(modelName)) {
                this.updateStatus('Loading cached model...', 50);
                this.session = this.modelCache.get(modelName);
                this.modelInfo = this.parseModelName(modelName);
                
                // Update UI
                this.elements.imageInput.disabled = false;
                this.elements.modelSize.textContent = 'Loaded (Cached)';
                this.elements.upscaleFactor.textContent = `${this.modelInfo.scale}x`;
                
                this.updateStatus('Cached model loaded!', 100);
                setTimeout(() => {
                    this.updateStatus('Model ready. Upload an image to enhance.', 0);
                }, 1000);
                return;
            }
            
            this.updateStatus('Loading model...', 10);
            this.elements.loadModelBtn.disabled = true;
            
            // Path to ONNX models (served by our development server)
            const modelPath = `/onnx_models/${modelName}.onnx`;
            
            this.updateStatus('Downloading model...', 30);
            
            // Configure ONNX Runtime Web paths and environment
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
            ort.env.logLevel = 'warning'; // Enable more detailed logging
            
            // Log ONNX Runtime version
            console.log('ONNX Runtime Web version:', ort.version || 'unknown');
            
            this.updateStatus('Detecting GPU capabilities...', 40);
            
            // Detect available execution providers with proper fallback
            // Temporarily force WebAssembly for debugging
            const executionProviders = ['wasm']; // await this.getOptimalExecutionProviders();
            
            this.updateStatus('Initializing model...', 60);
            
            // Add debugging for model loading
            console.log(`Loading model from: ${modelPath}`);
            console.log(`Execution providers: ${executionProviders.join(', ')}`);
            
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: executionProviders,
                graphOptimizationLevel: 'all',
                executionMode: 'sequential',
                enableMemPattern: true
            });
            
            // Verify session was created properly
            if (!this.session) {
                throw new Error('Session creation returned null/undefined');
            }
            
            console.log('Session created successfully:', {
                hasExecutionProviders: !!this.session.executionProviders,
                executionProvidersLength: this.session.executionProviders?.length,
                hasInputs: !!this.session.inputNames,
                hasOutputs: !!this.session.outputNames
            });
            
            // Cache the model
            this.modelCache.set(modelName, this.session);
            
            this.modelInfo = this.parseModelName(modelName);
            
            // Display which execution provider is being used
            let activeProvider = 'unknown';
            if (this.session && this.session.executionProviders && this.session.executionProviders.length > 0) {
                activeProvider = this.session.executionProviders[0];
            }
            console.log(`Using execution provider: ${activeProvider}`);
            console.log('Session object:', this.session);
            
            this.updateStatus(`Model loaded successfully! (${activeProvider})`, 100);
            
            // Update UI
            this.elements.imageInput.disabled = false;
            this.elements.modelSize.textContent = `Loaded (${activeProvider})`;
            this.elements.upscaleFactor.textContent = `${this.modelInfo.scale}x`;
            
            setTimeout(() => {
                this.updateStatus('Model ready. Upload an image to enhance.', 0);
            }, 2000);
            
        } catch (error) {
            console.error('Model loading error:', error);
            console.error('Error stack:', error.stack);
            this.updateStatus(`Error loading model: ${error.message}`, 0);
            this.elements.loadModelBtn.disabled = false;
        }
    }
    
    async getOptimalExecutionProviders() {
        /*
         * Detect and return optimal execution providers with proper fallback
         * Priority: WebGPU > WebGL > WebAssembly
         */
        const providers = [];
        
        try {
            // Check for WebGPU support (most performant)
            if ('gpu' in navigator) {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        providers.push('webgpu');
                        console.log('✅ WebGPU supported - using GPU acceleration');
                    }
                } catch (e) {
                    console.log('❌ WebGPU not available:', e.message);
                }
            }
            
            // Check for WebGL support (good performance)
            try {
                const canvas = document.createElement('canvas');
                const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                if (gl) {
                    providers.push('webgl');
                    console.log('✅ WebGL supported - using GPU acceleration');
                }
            } catch (e) {
                console.log('❌ WebGL not available:', e.message);
            }
            
            // WebAssembly fallback (always available)
            providers.push('wasm');
            console.log('✅ WebAssembly available - using CPU acceleration');
            
        } catch (error) {
            console.error('Error detecting execution providers:', error);
            // Fallback to just WebAssembly if detection fails
            return ['wasm'];
        }
        
        console.log(`Execution provider priority: ${providers.join(' > ')}`);
        return providers.length > 0 ? providers : ['wasm']; // Ensure we always return at least one provider
    }
    
    parseModelName(modelName) {
        const parts = modelName.split('_');
        const scaleMatch = parts.find(p => p.startsWith('s'));
        const scale = scaleMatch ? parseInt(scaleMatch.substring(1)) : 2;
        
        return {
            name: modelName,
            scale: scale,
            architecture: parts[0]
        };
    }
    
    async loadImage(file) {
        try {
            this.updateStatus('Loading image...', 20);
            
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            img.onload = () => {
                // Draw original image
                const canvas = this.elements.originalCanvas;
                const ctx = canvas.getContext('2d');
                
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                this.elements.originalInfo.textContent = `${img.width} × ${img.height}`;
                this.elements.processBtn.disabled = false;
                
                this.updateStatus('Image loaded. Ready to enhance.', 0);
            };
            
            img.onerror = () => {
                throw new Error('Failed to load image');
            };
            
            img.src = URL.createObjectURL(file);
            
        } catch (error) {
            console.error('Image loading error:', error);
            this.updateStatus(`Error loading image: ${error.message}`, 0);
        }
    }
    
    async processImage() {
        if (!this.session || this.isProcessing) return;
        
        this.isProcessing = true;
        this.elements.processBtn.disabled = true;
        
        try {
            const startTime = performance.now();
            const colorMode = this.elements.colorMode.value;
            
            this.updateStatus('Preparing image...', 10);
            
            // Get image data from canvas
            const canvas = this.elements.originalCanvas;
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            if (colorMode === 'color') {
                await this.processImageColor(imageData, startTime);
            } else {
                await this.processImageGrayscale(imageData, startTime);
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            this.updateStatus(`Error processing image: ${error.message}`, 0);
        } finally {
            this.isProcessing = false;
            this.elements.processBtn.disabled = false;
        }
    }
    
    async processImageGrayscale(imageData, startTime) {
        this.updateStatus('Converting to grayscale...', 30);
        
        // Convert to grayscale and normalize
        const inputTensor = this.preprocessImage(imageData);
        
        this.updateStatus('Running inference...', 50);
        
        // Run inference
        const feeds = { input: inputTensor };
        const results = await this.session.run(feeds);
        const outputTensor = results.output;
        
        this.updateStatus('Processing results...', 80);
        
        // Convert output back to image
        this.postprocessImage(outputTensor, 'grayscale');
        
        const endTime = performance.now();
        const processingTime = Math.round(endTime - startTime);
        
        this.elements.processingTime.textContent = `${processingTime}ms (Grayscale)`;
        this.elements.downloadBtn.disabled = false;
        
        this.updateStatus('Enhancement complete!', 100);
        
        setTimeout(() => {
            this.updateStatus('Ready for next image.', 0);
        }, 2000);
    }
    
    async processImageColor(imageData, startTime) {
        this.updateStatus('Extracting color channels...', 20);
        
        // Extract RGB channels
        const colorChannels = this.preprocessImageColor(imageData);
        
        this.updateStatus('Processing Red channel...', 30);
        const redResult = await this.session.run({ input: colorChannels.r });
        
        this.updateStatus('Processing Green channel...', 50);
        const greenResult = await this.session.run({ input: colorChannels.g });
        
        this.updateStatus('Processing Blue channel...', 70);
        const blueResult = await this.session.run({ input: colorChannels.b });
        
        this.updateStatus('Combining color channels...', 85);
        
        // Combine RGB results
        this.postprocessImageColor(redResult.output, greenResult.output, blueResult.output);
        
        const endTime = performance.now();
        const processingTime = Math.round(endTime - startTime);
        
        this.elements.processingTime.textContent = `${processingTime}ms (Color)`;
        this.elements.downloadBtn.disabled = false;
        
        this.updateStatus('Color enhancement complete!', 100);
        
        setTimeout(() => {
            this.updateStatus('Ready for next image.', 0);
        }, 2000);
    }
    
    preprocessImage(imageData) {
        const { width, height, data } = imageData;
        
        // For now, convert to grayscale to match paper's design
        // TODO: Add option for channel-wise color processing
        const grayscaleData = new Float32Array(width * height);
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Convert to grayscale using exact same formula as PIL convert('L')
            // This matches run.py: Image.open(input_file).convert('L')
            const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            grayscaleData[i / 4] = gray;
        }
        
        // Create tensor with shape [1, 1, height, width] - matching PyTorch unsqueeze(0)
        return new ort.Tensor('float32', grayscaleData, [1, 1, height, width]);
    }
    
    // Process color channels separately (higher quality but 3x slower)
    preprocessImageColor(imageData) {
        const { width, height, data } = imageData;
        
        // Extract RGB channels
        const rData = new Float32Array(width * height);
        const gData = new Float32Array(width * height);
        const bData = new Float32Array(width * height);
        
        for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            rData[pixelIndex] = data[i] / 255.0;     // R
            gData[pixelIndex] = data[i + 1] / 255.0; // G
            bData[pixelIndex] = data[i + 2] / 255.0; // B
        }
        
        return {
            r: new ort.Tensor('float32', rData, [1, 1, height, width]),
            g: new ort.Tensor('float32', gData, [1, 1, height, width]),
            b: new ort.Tensor('float32', bData, [1, 1, height, width])
        };
    }
    
    postprocessImage(outputTensor, mode = 'grayscale') {
        const [batch, channels, height, width] = outputTensor.dims;
        const data = outputTensor.data;
        
        // Create enhanced canvas
        const canvas = this.elements.enhancedCanvas;
        const ctx = canvas.getContext('2d');
        
        canvas.width = width;
        canvas.height = height;
        
        // Convert tensor data back to image data - matching run.py pipeline:
        // output_rgb = model(input_tensor).data[0].clamp(0, 1.).expand(3, -1, -1).permute(1, 2, 0).cpu().numpy() * 255.
        const imageData = ctx.createImageData(width, height);
        const pixels = imageData.data;
        
        for (let i = 0; i < data.length; i++) {
            // Clamp to [0, 1] range (matching .clamp(0, 1.) in run.py)
            const clampedValue = Math.min(1.0, Math.max(0.0, data[i]));
            
            // Scale to [0, 255] (matching * 255. in run.py)
            const pixelValue = Math.round(clampedValue * 255);
            
            const pixelIndex = i * 4;
            
            // Expand grayscale to RGB channels (matching .expand(3, -1, -1) in run.py)
            pixels[pixelIndex] = pixelValue;     // R
            pixels[pixelIndex + 1] = pixelValue; // G
            pixels[pixelIndex + 2] = pixelValue; // B
            pixels[pixelIndex + 3] = 255;        // A (full opacity)
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        this.elements.enhancedInfo.textContent = `${width} × ${height} (${mode})`;
    }
    
    postprocessImageColor(redTensor, greenTensor, blueTensor) {
        const [batch, channels, height, width] = redTensor.dims;
        
        // Create enhanced canvas
        const canvas = this.elements.enhancedCanvas;
        const ctx = canvas.getContext('2d');
        
        canvas.width = width;
        canvas.height = height;
        
        // Convert tensor data back to color image
        const imageData = ctx.createImageData(width, height);
        const pixels = imageData.data;
        
        for (let i = 0; i < redTensor.data.length; i++) {
            // Clamp each channel to [0, 1] range
            const r = Math.min(1.0, Math.max(0.0, redTensor.data[i]));
            const g = Math.min(1.0, Math.max(0.0, greenTensor.data[i]));
            const b = Math.min(1.0, Math.max(0.0, blueTensor.data[i]));
            
            // Scale to [0, 255]
            const pixelIndex = i * 4;
            pixels[pixelIndex] = Math.round(r * 255);     // R
            pixels[pixelIndex + 1] = Math.round(g * 255); // G
            pixels[pixelIndex + 2] = Math.round(b * 255); // B
            pixels[pixelIndex + 3] = 255;                 // A (full opacity)
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        this.elements.enhancedInfo.textContent = `${width} × ${height} (Color)`;
    }
    
    downloadImage() {
        const canvas = this.elements.enhancedCanvas;
        const link = document.createElement('a');
        
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            link.href = url;
            link.download = `enhanced_${this.modelInfo.name}_${Date.now()}.png`;
            link.click();
            URL.revokeObjectURL(url);
        }, 'image/png');
    }

    // Add method to clear cache if needed
    clearModelCache() {
        this.modelCache.clear();
        this.updateStatus('Model cache cleared', 0);
    }
}

// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const processor = new EdgeSRProcessor();
    
    // Add drag and drop functionality
    const imageInput = document.getElementById('imageInput');
    const imageContainer = document.querySelector('.image-container');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        imageContainer.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        imageContainer.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        imageContainer.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        imageContainer.classList.add('loading');
    }
    
    function unhighlight() {
        imageContainer.classList.remove('loading');
    }
    
    imageContainer.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            imageInput.files = files;
            processor.loadImage(files[0]);
        }
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case 'o':
                    e.preventDefault();
                    imageInput.click();
                    break;
                case 's':
                    e.preventDefault();
                    if (!processor.elements.downloadBtn.disabled) {
                        processor.downloadImage();
                    }
                    break;
                case 'Enter':
                    e.preventDefault();
                    if (!processor.elements.processBtn.disabled) {
                        processor.processImage();
                    }
                    break;
            }
        }
    });
    
    // Add performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log(`Page load time: ${Math.round(perfData.loadEventEnd - perfData.loadEventStart)}ms`);
            }, 0);
        });
    }
}); 