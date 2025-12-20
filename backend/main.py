from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor

# Interpolation methods mapping for image preprocessing
INTERPOLATION_METHODS = {
    'nearest': Image.Resampling.NEAREST,
    'bilinear': Image.Resampling.BILINEAR,
    'bicubic': Image.Resampling.BICUBIC,
    'lanczos': Image.Resampling.LANCZOS,
    'box': Image.Resampling.BOX,
    'hamming': Image.Resampling.HAMMING,
}

app = FastAPI(title="Trinetr Vision AI Visualization API")

# Load ImageNet labels
IMAGENET_LABELS = None
try:
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as response:
        labels = response.read().decode('utf-8').strip().split('\n')
        IMAGENET_LABELS = {i: label for i, label in enumerate(labels)}
except:
    # Fallback if network request fails
    IMAGENET_LABELS = {i: f"Class {i}" for i in range(1000)}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
loaded_models = {}

# Pydantic models for request bodies
class WeightUpdateRequest(BaseModel):
    filter_index: int
    weight_updates: dict

@app.get("/")
async def root():
    return {"message": "Trinetr Vision AI Visualization API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/preprocess/visualize")
async def visualize_preprocessing(
    file: UploadFile = File(...),
    target_size: int = 224,
    interpolation: str = "bilinear"
):
    """Visualize how an image is preprocessed (resized) for model input"""
    try:
        # Read and open original image
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = original_image.size
        
        # Get interpolation method
        if interpolation not in INTERPOLATION_METHODS:
            raise HTTPException(status_code=400, detail=f"Unknown interpolation: {interpolation}")
        
        interp_method = INTERPOLATION_METHODS[interpolation]
        
        # Resize image
        resized_image = original_image.resize((target_size, target_size), interp_method)
        
        # Convert resized image to base64 for frontend display
        buffer = io.BytesIO()
        resized_image.save(buffer, format='PNG')
        resized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Calculate scaling info
        scale_x = target_size / original_size[0]
        scale_y = target_size / original_size[1]
        
        # Compute pixel-level statistics
        original_array = np.array(original_image)
        resized_array = np.array(resized_image)
        
        return {
            "original_size": list(original_size),
            "target_size": [target_size, target_size],
            "scale_factor": [round(scale_x, 4), round(scale_y, 4)],
            "interpolation": interpolation,
            "resized_image": f"data:image/png;base64,{resized_base64}",
            "original_stats": {
                "mean": [float(original_array[:,:,i].mean()) for i in range(3)],
                "std": [float(original_array[:,:,i].std()) for i in range(3)],
                "min": [int(original_array[:,:,i].min()) for i in range(3)],
                "max": [int(original_array[:,:,i].max()) for i in range(3)],
            },
            "resized_stats": {
                "mean": [float(resized_array[:,:,i].mean()) for i in range(3)],
                "std": [float(resized_array[:,:,i].std()) for i in range(3)],
                "min": [int(resized_array[:,:,i].min()) for i in range(3)],
                "max": [int(resized_array[:,:,i].max()) for i in range(3)],
            },
            "available_interpolations": list(INTERPOLATION_METHODS.keys()),
            "interpolation_info": {
                "nearest": "Fastest, pixelated results. Each output pixel takes value from nearest input pixel.",
                "bilinear": "Smooth results using linear interpolation. Considers 4 nearest pixels.",
                "bicubic": "Smoother than bilinear, uses 16 nearest pixels. Good for photos.",
                "lanczos": "Highest quality, uses sinc function. Best for downscaling photos.",
                "box": "Simple averaging of pixels. Good for downscaling.",
                "hamming": "Similar to bilinear but with different weights. Less blurry edges.",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/transform")
async def transform_image(
    file: UploadFile = File(...),
    target_size: int = 224,
    interpolation: str = "bilinear",
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    red_shift: float = 0,
    green_shift: float = 0,
    blue_shift: float = 0,
    blur: float = 0,
    noise: float = 0,
    rotation: float = 0,
    flip_h: bool = False,
    flip_v: bool = False,
    occlusion_enabled: bool = False,
    occlusion_x: float = 50,
    occlusion_y: float = 50,
    occlusion_size: float = 20,
    use_custom_kernel: bool = False,
    custom_kernel: str = "[[1]]",
):
    """Transform an image with various adjustments and return both original and processed"""
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps
        import json
        
        # Read and open original image
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = original_image.size
        
        # Get interpolation method
        if interpolation not in INTERPOLATION_METHODS:
            interpolation = "bilinear"
        interp_method = INTERPOLATION_METHODS[interpolation]
        
        # Resize original for display (thumbnail)
        display_original = original_image.copy()
        display_original.thumbnail((224, 224), Image.Resampling.LANCZOS)
        
        # Start processing: resize first
        processed = original_image.resize((target_size, target_size), interp_method)
        
        # Apply rotation
        if rotation != 0:
            processed = processed.rotate(-rotation, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128, 128, 128))
        
        # Apply flips
        if flip_h:
            processed = processed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_v:
            processed = processed.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Apply brightness
        if brightness != 0:
            enhancer = ImageEnhance.Brightness(processed)
            processed = enhancer.enhance(1 + brightness / 100)
        
        # Apply contrast
        if contrast != 0:
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1 + contrast / 100)
        
        # Apply saturation
        if saturation != 0:
            enhancer = ImageEnhance.Color(processed)
            processed = enhancer.enhance(1 + saturation / 100)
        
        # Apply color shifts
        if red_shift != 0 or green_shift != 0 or blue_shift != 0:
            arr = np.array(processed, dtype=np.float32)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + red_shift * 2.55, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + green_shift * 2.55, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + blue_shift * 2.55, 0, 255)
            processed = Image.fromarray(arr.astype(np.uint8))
        
        # Apply blur
        if blur > 0:
            processed = processed.filter(ImageFilter.GaussianBlur(radius=blur))
        
        # Apply noise
        if noise > 0:
            arr = np.array(processed, dtype=np.float32)
            noise_arr = np.random.normal(0, noise * 2.55, arr.shape)
            arr = np.clip(arr + noise_arr, 0, 255)
            processed = Image.fromarray(arr.astype(np.uint8))
        
        # Apply occlusion
        if occlusion_enabled:
            arr = np.array(processed)
            ox = int((occlusion_x / 100) * target_size)
            oy = int((occlusion_y / 100) * target_size)
            size = int((occlusion_size / 100) * target_size)
            x1 = max(0, ox - size // 2)
            x2 = min(target_size, ox + size // 2)
            y1 = max(0, oy - size // 2)
            y2 = min(target_size, oy + size // 2)
            arr[y1:y2, x1:x2] = 128  # Gray occlusion
            processed = Image.fromarray(arr)
        
        # Apply custom convolution kernel using numpy
        if use_custom_kernel:
            try:
                kernel = json.loads(custom_kernel)
                kernel_array = np.array(kernel, dtype=np.float32)
                arr = np.array(processed, dtype=np.float32)
                
                # Simple convolution using numpy
                kh, kw = kernel_array.shape
                pad_h, pad_w = kh // 2, kw // 2
                
                # Apply kernel to each channel
                result = np.zeros_like(arr)
                for c in range(3):
                    channel = arr[:, :, c]
                    # Pad the image
                    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                    # Convolve
                    for i in range(arr.shape[0]):
                        for j in range(arr.shape[1]):
                            region = padded[i:i+kh, j:j+kw]
                            result[i, j, c] = np.sum(region * kernel_array)
                
                # Clip and convert back
                result = np.clip(result, 0, 255).astype(np.uint8)
                processed = Image.fromarray(result)
            except Exception as e:
                print(f"Custom kernel error: {e}")
        
        # Convert images to base64
        orig_buffer = io.BytesIO()
        display_original.save(orig_buffer, format='PNG')
        orig_base64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')
        
        proc_buffer = io.BytesIO()
        processed.save(proc_buffer, format='PNG')
        proc_base64 = base64.b64encode(proc_buffer.getvalue()).decode('utf-8')
        
        # Calculate stats
        scale_x = target_size / original_size[0]
        scale_y = target_size / original_size[1]
        
        orig_arr = np.array(display_original)
        proc_arr = np.array(processed)
        
        return {
            "original_size": list(original_size),
            "target_size": [target_size, target_size],
            "scale_factor": [round(scale_x, 4), round(scale_y, 4)],
            "interpolation": interpolation,
            "original_image": f"data:image/png;base64,{orig_base64}",
            "processed_image": f"data:image/png;base64,{proc_base64}",
            "original_stats": {
                "mean": [float(orig_arr[:,:,i].mean()) for i in range(3)],
                "std": [float(orig_arr[:,:,i].std()) for i in range(3)],
            },
            "processed_stats": {
                "mean": [float(proc_arr[:,:,i].mean()) for i in range(3)],
                "std": [float(proc_arr[:,:,i].std()) for i in range(3)],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
async def load_model(model_name: str = "resnet18"):
    """Load a pre-trained Vision AI model"""
    try:
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.eval()
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.eval()
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.eval()
        else:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")
        
        model_id = f"{model_name}_{len(loaded_models)}"
        loaded_models[model_id] = model
        
        # Get model architecture info
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layers.append({
                    "name": name,
                    "type": type(module).__name__
                })
        
        return {
            "model_id": model_id,
            "model_name": model_name,
            "layers": layers[:20]  # Limit to first 20 layers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/weight-structure")
async def get_weight_structure(model_id: str):
    """Get detailed weight structure information for a model"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = loaded_models[model_id]
    weight_structure = []
    total_params = 0
    total_trainable = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                "layer_name": name,
                "layer_type": type(module).__name__,
                "parameters": {}
            }
            
            # Get all parameters (weights, biases, etc.)
            for param_name, param in module.named_parameters(recurse=False):
                param_data = {
                    "shape": list(param.shape),
                    "numel": int(param.numel()),
                    "requires_grad": bool(param.requires_grad),
                    "dtype": str(param.dtype),
                    "min": float(param.data.min().item()) if param.numel() > 0 else 0.0,
                    "max": float(param.data.max().item()) if param.numel() > 0 else 0.0,
                    "mean": float(param.data.mean().item()) if param.numel() > 0 else 0.0,
                    "std": float(param.data.std().item()) if param.numel() > 0 else 0.0
                }
                layer_info["parameters"][param_name] = param_data
                total_params += param_data["numel"]
                if param_data["requires_grad"]:
                    total_trainable += param_data["numel"]
            
            # Get buffers (like running_mean, running_var for BatchNorm)
            buffers = {}
            for buffer_name, buffer in module.named_buffers(recurse=False):
                buffers[buffer_name] = {
                    "shape": list(buffer.shape),
                    "numel": int(buffer.numel()),
                    "dtype": str(buffer.dtype)
                }
            
            if buffers:
                layer_info["buffers"] = buffers
            
            # Add layer-specific information
            if isinstance(module, nn.Conv2d):
                layer_info["details"] = {
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size) if hasattr(module.kernel_size, '__iter__') else [module.kernel_size],
                    "stride": list(module.stride) if hasattr(module.stride, '__iter__') else [module.stride],
                    "padding": list(module.padding) if hasattr(module.padding, '__iter__') else [module.padding],
                    "has_bias": module.bias is not None
                }
            elif isinstance(module, nn.Linear):
                layer_info["details"] = {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "has_bias": module.bias is not None
                }
            elif isinstance(module, nn.BatchNorm2d):
                layer_info["details"] = {
                    "num_features": module.num_features,
                    "eps": module.eps,
                    "momentum": module.momentum,
                    "affine": module.affine,
                    "track_running_stats": module.track_running_stats
                }
            
            if layer_info["parameters"] or buffers:
                weight_structure.append(layer_info)
    
    return {
        "model_id": model_id,
        "total_parameters": total_params,
        "trainable_parameters": total_trainable,
        "non_trainable_parameters": total_params - total_trainable,
        "layers": weight_structure
    }

@app.get("/models/{model_id}/layers")
async def get_model_layers(model_id: str):
    """Get all layers of a loaded model"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = loaded_models[model_id]
    layers = []
    layer_map = {}
    
    # Build layer hierarchy
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_info = {
                "name": name,
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters())
            }
            
            # Get layer description
            layer_type = type(module).__name__
            if "Conv2d" in layer_type:
                layer_info["description"] = f"Convolutional layer with {module.out_channels} output channels"
                if hasattr(module, 'kernel_size'):
                    layer_info["description"] += f", kernel size {module.kernel_size}"
            elif "BatchNorm2d" in layer_type:
                layer_info["description"] = "Batch normalization layer - normalizes activations"
            elif "ReLU" in layer_type:
                layer_info["description"] = "ReLU activation - applies max(0, x) to introduce non-linearity"
            elif "MaxPool2d" in layer_type:
                layer_info["description"] = "Max pooling - reduces spatial dimensions"
            elif "AdaptiveAvgPool2d" in layer_type:
                layer_info["description"] = "Adaptive average pooling - global pooling"
            elif "Linear" in layer_type:
                layer_info["description"] = f"Fully connected layer - {module.out_features} outputs (FINAL LAYER - produces class probabilities)"
            else:
                layer_info["description"] = f"{layer_type} layer"
            
            layers.append(layer_info)
            layer_map[name] = layer_info
    
    # Determine actual execution order by tracing the model
    # Create a dummy input and trace which layers execute in order
    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        execution_order = {}
        execution_index = [0]  # Use list to allow modification in nested function
        
        # Hook to capture layer execution order
        def make_hook(name):
            def hook(module, input, output):
                if name not in execution_order:
                    execution_order[name] = execution_index[0]
                    execution_index[0] += 1
            return hook
        
        # Register hooks on all leaf modules
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Run forward pass to capture execution order
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Sort layers by their actual execution order
        def get_execution_order(layer):
            name = layer["name"]
            # Use actual execution order if available, otherwise fallback to name-based ordering
            if name in execution_order:
                return (execution_order[name], name)
            else:
                # Fallback: use name-based ordering
                name_lower = name.lower()
                layer_type = layer["type"]
                depth = name.count('.')
                
                # Check if VGG
                is_vgg = any('features' in l["name"] or 'classifier' in l["name"] for l in layers)
                
                if is_vgg:
                    if 'features' in name:
                        parts = name.split('.')
                        if len(parts) >= 2 and parts[0] == 'features':
                            try:
                                return (int(parts[1]), name)
                            except:
                                return (9999, name)
                    elif 'classifier' in name:
                        parts = name.split('.')
                        if len(parts) >= 2 and parts[0] == 'classifier':
                            try:
                                return (10000 + int(parts[1]), name)
                            except:
                                return (19999, name)
                    else:
                        if 'avgpool' in name_lower or 'adaptive' in name_lower:
                            return (5000, name)
                        return (9999, name)
                
                # ResNet fallback
                if '.' not in name:
                    if 'conv1' in name_lower:
                        return (0, name)
                    elif 'bn1' in name_lower:
                        return (1, name)
                    elif 'relu' in name_lower and '1' in name:
                        return (2, name)
                    elif 'maxpool' in name_lower:
                        return (3, name)
                    elif 'avgpool' in name_lower or 'adaptive' in name_lower:
                        return (9998, name)
                    elif 'fc' in name_lower or layer_type == "Linear":
                        return (9999, name)
                
                if "fc" in name_lower or layer_type == "Linear":
                    return (10000, name)
                return (depth * 1000, name)
        
        layers.sort(key=get_execution_order)
        
    except Exception as e:
        # Fallback to name-based ordering if tracing fails
        print(f"Warning: Could not trace execution order: {e}")
        is_vgg_model = any('features' in l["name"] or 'classifier' in l["name"] for l in layers)
        
        def get_execution_order(layer):
            name = layer["name"].lower()
            layer_type = layer["type"]
            full_name = layer["name"]
            
            if is_vgg_model:
                if 'features' in full_name:
                    parts = full_name.split('.')
                    if len(parts) >= 2 and parts[0] == 'features':
                        try:
                            idx = int(parts[1])
                            return (0, idx)
                        except:
                            return (0, 999)
                elif 'classifier' in full_name:
                    parts = full_name.split('.')
                    if len(parts) >= 2 and parts[0] == 'classifier':
                        try:
                            idx = int(parts[1])
                            return (2000, idx)
                        except:
                            return (2000, 999)
                else:
                    if 'avgpool' in name or 'adaptive' in name or layer_type == "AdaptiveAvgPool2d":
                        return (1000, 0)
                    return (1500, full_name)
            
            if '.' not in layer["name"]:
                if 'conv1' in name:
                    return (0, 0)
                elif 'bn1' in name:
                    return (0, 1)
                elif 'relu' in name and '1' in name:
                    return (0, 2)
                elif 'maxpool' in name:
                    return (0, 3)
                elif 'avgpool' in name or 'adaptive' in name:
                    return (999, 0)
                elif 'fc' in name or layer_type == "Linear":
                    return (999, 1)
                else:
                    return (0, 99)
            
            depth = layer["name"].count('.')
            if "fc" in name or layer_type == "Linear":
                return (1000, name)
            return (depth, name)
        
        layers.sort(key=get_execution_order)
    
    # Build connections and track previous layer
    for i in range(len(layers)):
        if i > 0:
            layers[i]["previous"] = layers[i - 1]["name"]
        if i < len(layers) - 1:
            layers[i]["next"] = layers[i + 1]["name"]
    
    return {"layers": layers, "architecture": "sequential"}

@app.post("/models/{model_id}/visualize/activations")
async def visualize_activations(
    model_id: str,
    layer_name: str,
    file: UploadFile = File(...)
):
    """Get activations for a specific layer"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Create feature extractor
        model = loaded_models[model_id]
        feature_extractor = create_feature_extractor(model, return_nodes=[layer_name])
        
        # Get activations
        with torch.no_grad():
            features = feature_extractor(input_tensor)
            activations = features[layer_name].squeeze(0).cpu().numpy()
        
        # Get layer info for input shape and previous layer
        previous_layer = None
        input_shape = None
        
        # Get all layers and determine actual execution order by tracing
        all_layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                all_layers.append({"name": name, "module": module})
        
        # Trace actual execution order using forward hooks
        execution_order = {}
        execution_index = [0]  # Use list to allow modification in nested function
        
        def make_hook(name):
            def hook(module, input, output):
                if name not in execution_order:
                    execution_order[name] = execution_index[0]
                    execution_index[0] += 1
            return hook
        
        hooks = []
        for l in all_layers:
            hook = l["module"].register_forward_hook(make_hook(l["name"]))
            hooks.append(hook)
        
        # Run forward pass to capture execution order
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Sort by actual execution order
        def get_layer_order(l):
            name = l["name"]
            if name in execution_order:
                return (execution_order[name], name)
            else:
                # Fallback ordering
                name_lower = name.lower()
                layer_type = type(l["module"]).__name__
                is_vgg = any('features' in ll["name"] or 'classifier' in ll["name"] for ll in all_layers)
                
                if is_vgg:
                    if 'features' in name:
                        parts = name.split('.')
                        if len(parts) >= 2 and parts[0] == 'features':
                            try:
                                return (int(parts[1]), name)
                            except:
                                return (9999, name)
                    elif 'classifier' in name:
                        parts = name.split('.')
                        if len(parts) >= 2 and parts[0] == 'classifier':
                            try:
                                return (10000 + int(parts[1]), name)
                            except:
                                return (19999, name)
                    else:
                        if 'avgpool' in name_lower or 'adaptive' in name_lower:
                            return (5000, name)
                        return (9999, name)
                
                if '.' not in name:
                    if 'conv1' in name_lower:
                        return (0, name)
                    elif 'bn1' in name_lower:
                        return (1, name)
                    elif 'relu' in name_lower and '1' in name:
                        return (2, name)
                    elif 'maxpool' in name_lower:
                        return (3, name)
                    elif 'avgpool' in name_lower or 'adaptive' in name_lower:
                        return (9998, name)
                    elif 'fc' in name_lower or layer_type == "Linear":
                        return (9999, name)
                
                depth = name.count('.')
                if "fc" in name_lower or layer_type == "Linear":
                    return (10000, name)
                return (depth * 1000, name)
        
        all_layers.sort(key=get_layer_order)
        
        # Find current layer index
        current_idx = None
        for idx, l in enumerate(all_layers):
            if l["name"] == layer_name:
                current_idx = idx
                break
        
        # Get previous layer and compute actual input shape
        if current_idx is not None and current_idx > 0:
            prev_layer_info = all_layers[current_idx - 1]
            previous_layer = prev_layer_info["name"]
            prev_module = prev_layer_info["module"]
            
            # Get actual input shape by running forward pass to previous layer
            try:
                # Create feature extractor for previous layer
                prev_feature_extractor = create_feature_extractor(model, return_nodes=[previous_layer])
                with torch.no_grad():
                    prev_features = prev_feature_extractor(input_tensor)
                    prev_output = prev_features[previous_layer]
                    if isinstance(prev_output, torch.Tensor):
                        actual_shape = list(prev_output.shape)
                        # Format as string with actual values
                        input_shape = f"[{', '.join(map(str, actual_shape))}]"
                    else:
                        input_shape = f"[{list(prev_output.shape)}]"
            except Exception as e:
                # Fallback to estimation if forward pass fails
                if hasattr(prev_module, 'out_channels'):
                    input_shape = f"[batch, {prev_module.out_channels}, H, W]"
                elif hasattr(prev_module, 'num_features'):
                    input_shape = f"[batch, {prev_module.num_features}, H, W]"
                elif type(prev_module).__name__ == "AdaptiveAvgPool2d":
                    input_shape = "[batch, flattened_features]"
                elif type(prev_module).__name__ == "MaxPool2d":
                    input_shape = "[batch, channels, H/2, W/2]"
                else:
                    input_shape = "[batch, channels, H, W]"
        elif current_idx == 0:
            # First layer - input is the image
            input_shape = f"[{', '.join(map(str, list(input_tensor.shape)))}]"
            previous_layer = "Input Image"
        
        # Normalize activations for visualization
        if len(activations.shape) == 3:  # Conv layer: [channels, height, width]
            num_channels, height, width = activations.shape
            
            # Create mean activation map
            activations_2d = np.mean(activations, axis=0)
            activations_2d = (activations_2d - activations_2d.min()) / (activations_2d.max() - activations_2d.min() + 1e-8)
            activations_2d = (activations_2d * 255).astype(np.uint8)
            
            # Get top channels for visualization
            channel_means = np.mean(activations, axis=(1, 2))
            top_channel_indices = np.argsort(channel_means)[-9:][::-1]  # Top 9 channels
            
            # Create sample channel visualizations (top 6)
            sample_channels = []
            for idx in top_channel_indices[:6]:  # Show top 6
                channel_data = activations[idx]
                channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
                channel_norm = (channel_norm * 255).astype(np.uint8)
                sample_channels.append({
                    "channel_index": int(idx),
                    "mean_activation": float(channel_means[idx]),
                    "data": channel_norm.tolist()
                })
            
            # Create all channels for full view
            all_channels = []
            for idx in range(num_channels):
                channel_data = activations[idx]
                channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
                channel_norm = (channel_norm * 255).astype(np.uint8)
                all_channels.append({
                    "channel_index": int(idx),
                    "mean_activation": float(channel_means[idx]),
                    "data": channel_norm.tolist()
                })
            
            return {
                "shape": list(activations.shape),
                "activations": activations_2d.tolist(),
                "channels": sample_channels,
                "all_channels": all_channels,
                "num_channels": int(num_channels),
                "min": float(activations.min()),
                "max": float(activations.max()),
                "mean": float(activations.mean()),
                "input_shape": input_shape,
                "previous_layer": previous_layer
            }
        elif len(activations.shape) == 1:  # Linear layer
            # Get top activations for visualization
            top_indices = np.argsort(activations)[-20:][::-1]  # Top 20
            
            top_activations = []
            for idx in top_indices:
                top_activations.append({
                    "index": int(idx),
                    "value": float(activations[idx]),
                    "label": IMAGENET_LABELS.get(int(idx), f"Class {idx}")
                })
            
            return {
                "shape": list(activations.shape),
                "activations": activations.tolist(),
                "top_activations": top_activations,
                "min": float(activations.min()),
                "max": float(activations.max()),
                "mean": float(activations.mean()),
                "is_linear": True,
                "input_shape": input_shape,
                "previous_layer": previous_layer
            }
        else:
            return {
                "shape": list(activations.shape),
                "activations": activations.tolist(),
                "min": float(activations.min()),
                "max": float(activations.max()),
                "mean": float(activations.mean())
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/weights/{layer_name}")
async def get_layer_weights(model_id: str, layer_name: str):
    """Get weights for a specific layer"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = loaded_models[model_id]
        
        # Get the layer module
        layer_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer_module = module
                break
        
        if layer_module is None:
            raise HTTPException(status_code=404, detail="Layer not found")
        
        # Check if it's a convolutional layer
        if isinstance(layer_module, nn.Conv2d):
            weights = layer_module.weight.data.cpu().numpy()
            bias = layer_module.bias.data.cpu().numpy() if layer_module.bias is not None else None
            
            # Get weight info
            out_channels, in_channels, kernel_h, kernel_w = weights.shape
            
            # Normalize weights for visualization (per filter)
            normalized_weights = []
            for i in range(min(out_channels, 64)):  # Limit to first 64 filters
                filter_weights = weights[i]
                # Normalize across all input channels
                filter_flat = filter_weights.flatten()
                min_val = filter_flat.min()
                max_val = filter_flat.max()
                range_val = max_val - min_val if max_val != min_val else 1
                
                normalized = (filter_weights - min_val) / range_val
                normalized_uint8 = (normalized * 255).astype(np.uint8)
                
                normalized_weights.append({
                    "filter_index": int(i),
                    "weights": normalized_uint8.tolist(),
                    "raw_weights": filter_weights.tolist(),  # Include raw weights
                    "raw_min": float(min_val),
                    "raw_max": float(max_val),
                    "raw_mean": float(filter_flat.mean()),
                    "raw_std": float(filter_flat.std())
                })
            
            return {
                "layer_name": layer_name,
                "layer_type": "Conv2d",
                "shape": list(weights.shape),
                "out_channels": int(out_channels),
                "in_channels": int(in_channels),
                "kernel_size": [int(kernel_h), int(kernel_w)],
                "filters": normalized_weights,
                "bias": bias.tolist() if bias is not None else None,
                "weight_stats": {
                    "min": float(weights.min()),
                    "max": float(weights.max()),
                    "mean": float(weights.mean()),
                    "std": float(weights.std())
                }
            }
        elif isinstance(layer_module, nn.Linear):
            weights = layer_module.weight.data.cpu().numpy()
            bias = layer_module.bias.data.cpu().numpy() if layer_module.bias is not None else None
            
            return {
                "layer_name": layer_name,
                "layer_type": "Linear",
                "shape": list(weights.shape),
                "in_features": int(weights.shape[1]),
                "out_features": int(weights.shape[0]),
                "weights": weights.tolist(),
                "bias": bias.tolist() if bias is not None else None,
                "weight_stats": {
                    "min": float(weights.min()),
                    "max": float(weights.max()),
                    "mean": float(weights.mean()),
                    "std": float(weights.std())
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Layer type not supported for weight visualization")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_id}/weights/{layer_name}/update")
async def update_layer_weights(
    model_id: str,
    layer_name: str,
    request: WeightUpdateRequest = Body(...)
):
    """Update weights for a specific filter in a layer (for tinkering)"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = loaded_models[model_id]
        
        # Get the layer module
        layer_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer_module = module
                break
        
        if layer_module is None:
            raise HTTPException(status_code=404, detail="Layer not found")
        
        if isinstance(layer_module, nn.Conv2d):
            # Get current weights
            weights = layer_module.weight.data
            
            filter_index = request.filter_index
            weight_updates = request.weight_updates
            
            # Update specific filter weights
            if filter_index >= weights.shape[0]:
                raise HTTPException(status_code=400, detail="Filter index out of range")
            
            # Apply updates (scale, shift, etc.)
            if "scale" in weight_updates:
                weights[filter_index] *= weight_updates["scale"]
            if "shift" in weight_updates:
                weights[filter_index] += weight_updates["shift"]
            if "multiply" in weight_updates:
                weights[filter_index] *= weight_updates["multiply"]
            
            # Update the layer
            layer_module.weight.data = weights
            
            new_stats = {
                "min": float(weights[filter_index].min().item()),
                "max": float(weights[filter_index].max().item()),
                "mean": float(weights[filter_index].mean().item())
            }
            
            return {
                "status": "success",
                "message": f"Updated filter {filter_index} in layer {layer_name}",
                "new_stats": new_stats
            }
        else:
            raise HTTPException(status_code=400, detail="Only Conv2d layers support weight updates")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_id}/predict")
async def predict(model_id: str, file: UploadFile = File(...)):
    """Get model prediction for an image with final layer activations"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Get prediction and final layer activations
        model = loaded_models[model_id]
        
        # Find the final fully connected layer name
        final_layer_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'fc' in name.lower():
                final_layer_name = name
            elif isinstance(module, nn.Linear):
                # For models where fc might be named differently
                final_layer_name = name
        
        with torch.no_grad():
            # Get raw logits from final layer
            if final_layer_name:
                feature_extractor = create_feature_extractor(model, return_nodes=[final_layer_name])
                features = feature_extractor(input_tensor)
                final_activations = features[final_layer_name].squeeze(0).cpu().numpy()
            else:
                final_activations = None
            
            # Get full model output
            output = model(input_tensor)
            logits = output[0].cpu().numpy()
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top10_prob, top10_idx = torch.topk(probabilities, 10)
        
        # Get predictions with labels
        results = []
        for prob, idx in zip(top10_prob, top10_idx):
            class_id = int(idx.item())
            label = IMAGENET_LABELS.get(class_id, f"Class {class_id}")
            results.append({
                "class_id": class_id,
                "label": label,
                "probability": float(prob.item()),
                "logit": float(logits[class_id])
            })
        
        return {
            "predictions": results,
            "final_layer_activations": final_activations.tolist() if final_activations is not None else None,
            "final_layer_name": final_layer_name,
            "logits": logits.tolist(),
            "top_prediction": {
                "class_id": int(top10_idx[0].item()),
                "label": IMAGENET_LABELS.get(int(top10_idx[0].item()), f"Class {int(top10_idx[0].item())}"),
                "probability": float(top10_prob[0].item()),
                "logit": float(logits[int(top10_idx[0].item())])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== TRANSFORMER MODELS ==============

# Storage for transformer models
loaded_transformers = {}

# Transformer model configurations
TRANSFORMER_MODELS = {
    'text': {
        'bert-base': {
            'hf_name': 'bert-base-uncased',
            'type': 'encoder',
            'description': 'BERT base model for text understanding'
        },
        'distilbert': {
            'hf_name': 'distilbert-base-uncased',
            'type': 'encoder',
            'description': 'Smaller, faster BERT'
        },
        'gpt2': {
            'hf_name': 'gpt2',
            'type': 'decoder',
            'description': 'GPT-2 for text generation'
        },
        'roberta': {
            'hf_name': 'roberta-base',
            'type': 'encoder',
            'description': 'Robustly optimized BERT'
        },
    },
    'image': {
        'vit-base': {
            'hf_name': 'google/vit-base-patch16-224',
            'type': 'encoder',
            'description': 'Vision Transformer for image classification'
        },
        'deit-small': {
            'hf_name': 'facebook/deit-small-patch16-224',
            'type': 'encoder',
            'description': 'Data-efficient Image Transformer'
        },
        'swin-tiny': {
            'hf_name': 'microsoft/swin-tiny-patch4-window7-224',
            'type': 'encoder',
            'description': 'Shifted Window Transformer'
        },
        'clip-vit': {
            'hf_name': 'openai/clip-vit-base-patch16',
            'type': 'encoder',
            'description': 'CLIP Vision Transformer'
        },
    }
}

@app.post("/transformers/load")
async def load_transformer(model_name: str, model_type: str = "text"):
    """Load a transformer model"""
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        
        if model_type not in TRANSFORMER_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        if model_name not in TRANSFORMER_MODELS[model_type]:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        config = TRANSFORMER_MODELS[model_type][model_name]
        hf_name = config['hf_name']
        
        model_id = f"{model_name}_{len(loaded_transformers)}"
        
        if model_type == 'text':
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            # GPT-2 and similar models don't have a pad token by default
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(hf_name, output_attentions=True, output_hidden_states=True)
            model.eval()
            
            loaded_transformers[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'type': model_type,
                'model_name': model_name,
                'config': config
            }
        else:  # image
            feature_extractor = AutoFeatureExtractor.from_pretrained(hf_name)
            try:
                model = AutoModelForImageClassification.from_pretrained(hf_name, output_attentions=True, output_hidden_states=True)
            except:
                model = AutoModel.from_pretrained(hf_name, output_attentions=True, output_hidden_states=True)
            model.eval()
            
            loaded_transformers[model_id] = {
                'model': model,
                'feature_extractor': feature_extractor,
                'type': model_type,
                'model_name': model_name,
                'config': config
            }
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        num_layers = len(model.encoder.layer) if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer') else 12
        
        return {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "hf_name": hf_name,
            "num_parameters": num_params,
            "num_layers": num_layers,
            "description": config['description']
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="transformers library not installed. Run: pip install transformers")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextInferenceRequest(BaseModel):
    text: str
    generate_tokens: int = 0  # Number of tokens to generate (0 = just encode)

@app.post("/transformers/{model_id}/infer")
async def transformer_text_inference(model_id: str, request: TextInferenceRequest):
    """Run inference on a text transformer model"""
    if model_id not in loaded_transformers:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = loaded_transformers[model_id]
    if model_data['type'] != 'text':
        raise HTTPException(status_code=400, detail="This endpoint is for text models")
    
    try:
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        model_name = model_data.get('model_name', '')
        config = model_data.get('config', {})
        is_decoder = config.get('type') == 'decoder' or 'gpt' in model_name.lower()
        
        # Tokenize input
        inputs = tokenizer(request.text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract attention weights - handle different model architectures
        attention_layers = []
        attentions = getattr(outputs, 'attentions', None)
        if attentions:
            for layer_idx, attention in enumerate(attentions):
                if attention is not None:
                    num_heads = min(attention.shape[1], 12)
                    for head_idx in range(num_heads):
                        attention_layers.append({
                            "layer": layer_idx,
                            "head": head_idx,
                            "attention_weights": attention[0, head_idx].cpu().numpy().tolist(),
                            "tokens": tokens
                        })
        
        # Extract Q, K, V weight info and compute actual QKV values
        qkv_info = None
        attention_flow = []  # Layer-by-layer flow data
        kv_cache_info = None
        
        try:
            hidden_size = model.config.hidden_size
            num_attention_heads = model.config.num_attention_heads
            num_layers = model.config.num_hidden_layers
            head_dim = hidden_size // num_attention_heads
            seq_len = len(tokens)
            
            # Get hidden states for computing actual QKV values
            all_hidden_states = getattr(outputs, 'hidden_states', None)
            all_attentions = getattr(outputs, 'attentions', None)
            
            # KV Cache info (theoretical for this sequence)
            kv_cache_info = {
                "enabled": True,
                "shape": {
                    "keys": [num_layers, seq_len, num_attention_heads, head_dim],
                    "values": [num_layers, seq_len, num_attention_heads, head_dim]
                },
                "size_per_token_bytes": num_layers * 2 * num_attention_heads * head_dim * 4,  # float32
                "total_size_bytes": num_layers * 2 * seq_len * num_attention_heads * head_dim * 4,
                "total_size_mb": round(num_layers * 2 * seq_len * num_attention_heads * head_dim * 4 / (1024 * 1024), 4)
            }
            
            # Get the first encoder/decoder layer to extract QKV info
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                # BERT-style
                layers = model.encoder.layer
                first_layer = layers[0]
                attn = first_layer.attention.self
                
                qkv_info = {
                    "hidden_size": hidden_size,
                    "num_heads": num_attention_heads,
                    "head_dim": head_dim,
                    "num_layers": num_layers,
                    "query_weight_shape": list(attn.query.weight.shape),
                    "key_weight_shape": list(attn.key.weight.shape),
                    "value_weight_shape": list(attn.value.weight.shape),
                    "query_bias": attn.query.bias is not None,
                    "key_bias": attn.key.bias is not None,
                    "value_bias": attn.value.bias is not None,
                    "query_weight_sample": attn.query.weight[:8, :8].detach().cpu().numpy().tolist(),
                    "key_weight_sample": attn.key.weight[:8, :8].detach().cpu().numpy().tolist(),
                    "value_weight_sample": attn.value.weight[:8, :8].detach().cpu().numpy().tolist(),
                }
                
                # Compute actual Q, K, V values for each layer
                if all_hidden_states:
                    for layer_idx, layer in enumerate(layers):
                        if layer_idx >= len(all_hidden_states) - 1:
                            break
                        
                        layer_attn = layer.attention.self
                        hidden_state = all_hidden_states[layer_idx]  # [batch, seq, hidden]
                        
                        # Compute Q, K, V
                        with torch.no_grad():
                            Q = layer_attn.query(hidden_state)  # [batch, seq, hidden]
                            K = layer_attn.key(hidden_state)
                            V = layer_attn.value(hidden_state)
                            
                            # Reshape for multi-head: [batch, seq, num_heads, head_dim]
                            batch_size = Q.shape[0]
                            Q_heads = Q.view(batch_size, -1, num_attention_heads, head_dim).transpose(1, 2)
                            K_heads = K.view(batch_size, -1, num_attention_heads, head_dim).transpose(1, 2)
                            V_heads = V.view(batch_size, -1, num_attention_heads, head_dim).transpose(1, 2)
                        
                        # Get attention weights for this layer
                        layer_attention = all_attentions[layer_idx] if all_attentions and layer_idx < len(all_attentions) else None
                        
                        layer_flow = {
                            "layer": layer_idx,
                            "input_shape": list(hidden_state.shape),
                            "heads": []
                        }
                        
                        for head_idx in range(min(num_attention_heads, 12)):
                            head_data = {
                                "head": head_idx,
                                "Q_shape": [seq_len, head_dim],
                                "K_shape": [seq_len, head_dim],
                                "V_shape": [seq_len, head_dim],
                                # Sample values for first 5 tokens, first 8 dims
                                "Q_sample": Q_heads[0, head_idx, :5, :8].detach().cpu().numpy().tolist(),
                                "K_sample": K_heads[0, head_idx, :5, :8].detach().cpu().numpy().tolist(),
                                "V_sample": V_heads[0, head_idx, :5, :8].detach().cpu().numpy().tolist(),
                                # Statistics
                                "Q_stats": {
                                    "mean": float(Q_heads[0, head_idx].mean().item()),
                                    "std": float(Q_heads[0, head_idx].std().item()),
                                    "min": float(Q_heads[0, head_idx].min().item()),
                                    "max": float(Q_heads[0, head_idx].max().item())
                                },
                                "K_stats": {
                                    "mean": float(K_heads[0, head_idx].mean().item()),
                                    "std": float(K_heads[0, head_idx].std().item()),
                                    "min": float(K_heads[0, head_idx].min().item()),
                                    "max": float(K_heads[0, head_idx].max().item())
                                },
                                "V_stats": {
                                    "mean": float(V_heads[0, head_idx].mean().item()),
                                    "std": float(V_heads[0, head_idx].std().item()),
                                    "min": float(V_heads[0, head_idx].min().item()),
                                    "max": float(V_heads[0, head_idx].max().item())
                                },
                            }
                            
                            # Add attention weights for this head
                            if layer_attention is not None:
                                head_data["attention_weights"] = layer_attention[0, head_idx, :10, :10].cpu().numpy().tolist()
                            
                            layer_flow["heads"].append(head_data)
                        
                        # Output hidden state stats
                        if layer_idx + 1 < len(all_hidden_states):
                            output_hidden = all_hidden_states[layer_idx + 1]
                            layer_flow["output_stats"] = {
                                "mean": float(output_hidden.mean().item()),
                                "std": float(output_hidden.std().item())
                            }
                        
                        attention_flow.append(layer_flow)
                        
                        # Limit to first 6 layers for performance
                        if layer_idx >= 5:
                            break
                
            elif hasattr(model, 'h'):  # GPT-2 style
                first_layer = model.h[0]
                attn = first_layer.attn
                
                qkv_info = {
                    "hidden_size": hidden_size,
                    "num_heads": num_attention_heads,
                    "head_dim": head_dim,
                    "num_layers": num_layers,
                    "combined_qkv": True,
                    "c_attn_weight_shape": list(attn.c_attn.weight.shape) if hasattr(attn, 'c_attn') else None,
                    "qkv_weight_sample": attn.c_attn.weight[:8, :24].detach().cpu().numpy().tolist() if hasattr(attn, 'c_attn') else None,
                }
                
                # Compute actual Q, K, V for GPT-2 style
                if all_hidden_states:
                    for layer_idx, layer in enumerate(model.h):
                        if layer_idx >= len(all_hidden_states) - 1:
                            break
                        
                        hidden_state = all_hidden_states[layer_idx]
                        layer_attn = layer.attn
                        
                        with torch.no_grad():
                            # GPT-2 combines QKV in c_attn
                            qkv = layer_attn.c_attn(hidden_state)
                            Q, K, V = qkv.split(hidden_size, dim=-1)
                            
                            batch_size = Q.shape[0]
                            Q_heads = Q.view(batch_size, -1, num_attention_heads, head_dim).transpose(1, 2)
                            K_heads = K.view(batch_size, -1, num_attention_heads, head_dim).transpose(1, 2)
                            V_heads = V.view(batch_size, -1, num_attention_heads, head_dim).transpose(1, 2)
                        
                        layer_attention = all_attentions[layer_idx] if all_attentions and layer_idx < len(all_attentions) else None
                        
                        layer_flow = {
                            "layer": layer_idx,
                            "input_shape": list(hidden_state.shape),
                            "heads": []
                        }
                        
                        for head_idx in range(min(num_attention_heads, 12)):
                            head_data = {
                                "head": head_idx,
                                "Q_shape": [seq_len, head_dim],
                                "K_shape": [seq_len, head_dim],
                                "V_shape": [seq_len, head_dim],
                                "Q_sample": Q_heads[0, head_idx, :5, :8].detach().cpu().numpy().tolist(),
                                "K_sample": K_heads[0, head_idx, :5, :8].detach().cpu().numpy().tolist(),
                                "V_sample": V_heads[0, head_idx, :5, :8].detach().cpu().numpy().tolist(),
                                "Q_stats": {
                                    "mean": float(Q_heads[0, head_idx].mean().item()),
                                    "std": float(Q_heads[0, head_idx].std().item()),
                                    "min": float(Q_heads[0, head_idx].min().item()),
                                    "max": float(Q_heads[0, head_idx].max().item())
                                },
                                "K_stats": {
                                    "mean": float(K_heads[0, head_idx].mean().item()),
                                    "std": float(K_heads[0, head_idx].std().item()),
                                    "min": float(K_heads[0, head_idx].min().item()),
                                    "max": float(K_heads[0, head_idx].max().item())
                                },
                                "V_stats": {
                                    "mean": float(V_heads[0, head_idx].mean().item()),
                                    "std": float(V_heads[0, head_idx].std().item()),
                                    "min": float(V_heads[0, head_idx].min().item()),
                                    "max": float(V_heads[0, head_idx].max().item())
                                },
                            }
                            
                            if layer_attention is not None:
                                head_data["attention_weights"] = layer_attention[0, head_idx, :10, :10].cpu().numpy().tolist()
                            
                            layer_flow["heads"].append(head_data)
                        
                        if layer_idx + 1 < len(all_hidden_states):
                            output_hidden = all_hidden_states[layer_idx + 1]
                            layer_flow["output_stats"] = {
                                "mean": float(output_hidden.mean().item()),
                                "std": float(output_hidden.std().item())
                            }
                        
                        attention_flow.append(layer_flow)
                        
                        if layer_idx >= 5:
                            break
                            
        except Exception as e:
            import traceback
            print(f"Could not extract QKV info: {e}")
            traceback.print_exc()
        
        # Extract embeddings (last hidden state)
        embeddings = []
        last_hidden = getattr(outputs, 'last_hidden_state', None)
        if last_hidden is not None:
            embeddings = last_hidden[0].cpu().numpy().tolist()
        
        # Get hidden states if available
        hidden_states = None
        hs = getattr(outputs, 'hidden_states', None)
        if hs:
            hidden_states = [h[0].cpu().numpy().tolist() for h in hs[-3:]]
        
        # For decoder models (GPT-2), do next token prediction
        generation_steps = []
        generated_tokens = []
        
        if is_decoder and request.generate_tokens > 0:
            current_ids = input_ids.clone()
            
            for step in range(request.generate_tokens):
                with torch.no_grad():
                    # Get model output
                    step_outputs = model(current_ids, output_attentions=True)
                    
                    # Get logits for the last token position
                    # For GPT-2 base model, we need to use the hidden states with lm_head
                    # But since we loaded AutoModel, we need to compute logits differently
                    last_hidden_state = step_outputs.last_hidden_state
                    
                    # Get the embedding matrix (word embeddings) to compute logits
                    if hasattr(model, 'wte'):  # GPT-2
                        lm_head = model.wte.weight
                    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
                        lm_head = model.embeddings.word_embeddings.weight
                    else:
                        # Fallback: just show the process without actual generation
                        break
                    
                    # Compute logits: hidden_state @ embedding_matrix.T
                    last_position_hidden = last_hidden_state[0, -1, :]  # [hidden_size]
                    logits = torch.matmul(last_position_hidden, lm_head.T)  # [vocab_size]
                    
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Get top 10 predictions
                    top_probs, top_indices = torch.topk(probs, 10)
                    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
                    
                    # Select the most likely token (greedy)
                    next_token_id = top_indices[0].item()
                    next_token = tokenizer.decode([next_token_id])
                    
                    # Store step info
                    generation_steps.append({
                        "step": step + 1,
                        "input_so_far": tokenizer.decode(current_ids[0]),
                        "top_predictions": [
                            {"token": tok, "probability": float(prob.item())}
                            for tok, prob in zip(top_tokens, top_probs)
                        ],
                        "selected_token": next_token,
                        "selected_token_id": next_token_id,
                        "logits_stats": {
                            "min": float(logits.min().item()),
                            "max": float(logits.max().item()),
                            "mean": float(logits.mean().item()),
                        }
                    })
                    
                    generated_tokens.append(next_token)
                    
                    # Append to input for next iteration
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]])], dim=1)
        
        return {
            "input_tokens": tokens,
            "attention_layers": attention_layers,
            "embeddings": embeddings,
            "hidden_states": hidden_states,
            "output_tokens": tokens,
            "is_decoder": is_decoder,
            "generation_steps": generation_steps,
            "generated_text": "".join(generated_tokens) if generated_tokens else None,
            "full_text": request.text + "".join(generated_tokens) if generated_tokens else request.text,
            "qkv_info": qkv_info,
            "attention_flow": attention_flow,
            "kv_cache_info": kv_cache_info
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transformers/{model_id}/infer-image")
async def transformer_image_inference(model_id: str, file: UploadFile = File(...)):
    """Run inference on an image transformer model"""
    if model_id not in loaded_transformers:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = loaded_transformers[model_id]
    if model_data['type'] != 'image':
        raise HTTPException(status_code=400, detail="This endpoint is for image models")
    
    try:
        model = model_data['model']
        feature_extractor = model_data['feature_extractor']
        
        # Load and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors='pt')
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract predictions if available
        predictions = []
        logits = getattr(outputs, 'logits', None)
        if logits is not None:
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            k = min(5, probs.shape[0])
            top_prob, top_idx = torch.topk(probs, k)
            
            # Try to get labels from model config, but prefer ImageNet labels if available
            id2label = getattr(model.config, 'id2label', None)
            
            for prob, idx in zip(top_prob, top_idx):
                idx_val = idx.item()
                
                # First try ImageNet labels (most vision models use these)
                if IMAGENET_LABELS and idx_val < 1000:
                    label = IMAGENET_LABELS.get(idx_val, None)
                    if label and not label.startswith("LABEL_"):
                        predictions.append({
                            "label": label,
                            "probability": float(prob.item())
                        })
                        continue
                
                # Fall back to model's id2label
                if id2label:
                    label = id2label.get(idx_val, f"Class {idx_val}")
                    # Skip generic labels, use class index instead
                    if label.startswith("LABEL_"):
                        label = IMAGENET_LABELS.get(idx_val, f"Class {idx_val}") if IMAGENET_LABELS else f"Class {idx_val}"
                else:
                    label = IMAGENET_LABELS.get(idx_val, f"Class {idx_val}") if IMAGENET_LABELS else f"Class {idx_val}"
                
                predictions.append({
                    "label": label,
                    "probability": float(prob.item())
                })
        
        # Extract attention weights - handle different model architectures
        attention_layers = []
        attentions = getattr(outputs, 'attentions', None)
        if attentions:
            for layer_idx, attention in enumerate(attentions):
                if attention is not None:
                    num_heads = min(attention.shape[1], 12)
                    for head_idx in range(num_heads):
                        # For ViT, create patch position labels
                        seq_len = attention.shape[-1]
                        tokens = ["[CLS]"] + [f"P{i}" for i in range(seq_len - 1)]
                        attention_layers.append({
                            "layer": layer_idx,
                            "head": head_idx,
                            "attention_weights": attention[0, head_idx].cpu().numpy().tolist(),
                            "tokens": tokens
                        })
        
        # Extract embeddings
        embeddings = []
        last_hidden = getattr(outputs, 'last_hidden_state', None)
        if last_hidden is not None:
            embeddings = last_hidden[0].cpu().numpy().tolist()
        
        # Extract Q, K, V weight info from ViT model
        qkv_info = None
        try:
            if hasattr(model, 'vit') and hasattr(model.vit, 'encoder'):
                # ViT model structure
                first_layer = model.vit.encoder.layer[0]
                attn = first_layer.attention.attention
                hidden_size = model.config.hidden_size
                num_attention_heads = model.config.num_attention_heads
                head_dim = hidden_size // num_attention_heads
                
                qkv_info = {
                    "hidden_size": hidden_size,
                    "num_heads": num_attention_heads,
                    "head_dim": head_dim,
                    "query_weight_shape": list(attn.query.weight.shape),
                    "key_weight_shape": list(attn.key.weight.shape),
                    "value_weight_shape": list(attn.value.weight.shape),
                    "query_bias": attn.query.bias is not None,
                    "key_bias": attn.key.bias is not None,
                    "value_bias": attn.value.bias is not None,
                    # Sample weights (first 8x8 block for visualization)
                    "query_weight_sample": attn.query.weight[:8, :8].detach().cpu().numpy().tolist(),
                    "key_weight_sample": attn.key.weight[:8, :8].detach().cpu().numpy().tolist(),
                    "value_weight_sample": attn.value.weight[:8, :8].detach().cpu().numpy().tolist(),
                }
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                # Alternative ViT structure
                first_layer = model.encoder.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    attn = first_layer.self_attn
                    hidden_size = model.config.hidden_size
                    num_attention_heads = model.config.num_attention_heads
                    head_dim = hidden_size // num_attention_heads
                    
                    qkv_info = {
                        "hidden_size": hidden_size,
                        "num_heads": num_attention_heads,
                        "head_dim": head_dim,
                    }
        except Exception as e:
            print(f"Could not extract QKV info for image model: {e}")
        
        return {
            "predictions": predictions,
            "attention_layers": attention_layers,
            "embeddings": embeddings,
            "input_tokens": ["[CLS]"] + [f"Patch {i}" for i in range(196)],  # Standard ViT has 196 patches
            "qkv_info": qkv_info
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transformers/{model_id}/architecture")
async def get_transformer_architecture(model_id: str):
    """Get the architecture of a transformer model"""
    if model_id not in loaded_transformers:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = loaded_transformers[model_id]
    model = model_data['model']
    
    layers = []
    
    def get_layer_info(name, module):
        layer_info = {
            "name": name,
            "type": module.__class__.__name__,
            "params": sum(p.numel() for p in module.parameters()),
        }
        
        # Add specific info based on layer type
        if hasattr(module, 'in_features'):
            layer_info['in_features'] = module.in_features
            layer_info['out_features'] = module.out_features
        if hasattr(module, 'num_attention_heads'):
            layer_info['num_heads'] = module.num_attention_heads
        if hasattr(module, 'hidden_size'):
            layer_info['hidden_size'] = module.hidden_size
            
        return layer_info
    
    for name, module in model.named_modules():
        if name and not any(child for child in module.children()):
            layers.append(get_layer_info(name, module))
    
    return {
        "model_id": model_id,
        "model_name": model_data['model_name'],
        "type": model_data['type'],
        "layers": layers,
        "total_params": sum(p.numel() for p in model.parameters())
    }

# ============== DIFFUSION MODELS ==============

# Storage for diffusion models
loaded_diffusion_models = {}

# Diffusion model configurations
DIFFUSION_MODELS = {
    'stable-diffusion-v1-4': {
        'hf_name': 'CompVis/stable-diffusion-v1-4',
        'description': 'Stable Diffusion v1.4 - Text-to-image generation',
        'requires_safety_checker': True
    },
    'stable-diffusion-v1-5': {
        'hf_name': 'runwayml/stable-diffusion-v1-5',
        'description': 'Stable Diffusion v1.5 - Improved version',
        'requires_safety_checker': True
    },
    'stable-diffusion-2-1': {
        'hf_name': 'stabilityai/stable-diffusion-2-1',
        'description': 'Stable Diffusion 2.1 - Latest stable version',
        'requires_safety_checker': True
    },
}

class DiffusionGenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    return_intermediates: bool = True  # Return intermediate steps for visualization

@app.post("/diffusion/load")
async def load_diffusion_model(model_name: str):
    """Load a diffusion model"""
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        
        if model_name not in DIFFUSION_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        config = DIFFUSION_MODELS[model_name]
        hf_name = config['hf_name']
        
        model_id = f"{model_name}_{len(loaded_diffusion_models)}"
        
        # Load the pipeline
        # Use CPU by default, but allow GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                hf_name,
                torch_dtype=dtype,
                safety_checker=None,  # Disable safety checker for visualization
                requires_safety_checker=False
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()  # Reduce memory usage
        except Exception as e:
            # Fallback: try loading without safety checker
            pipe = StableDiffusionPipeline.from_pretrained(
                hf_name,
                torch_dtype=dtype
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        
        loaded_diffusion_models[model_id] = {
            'pipeline': pipe,
            'model_name': model_name,
            'config': config,
            'device': device
        }
        
        return {
            "model_id": model_id,
            "model_name": model_name,
            "device": device,
            "description": config['description']
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="diffusers library not installed. Install with: pip install diffusers accelerate"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/diffusion/{model_id}/generate")
async def generate_diffusion_image(model_id: str, request: DiffusionGenerateRequest):
    """Generate an image using diffusion model with step-by-step visualization"""
    if model_id not in loaded_diffusion_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler
        
        model_data = loaded_diffusion_models[model_id]
        pipe = model_data['pipeline']
        device = model_data['device']
        
        # Set seed for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=device).manual_seed(request.seed)
        
        # Store intermediate latents for visualization
        intermediate_steps = []
        intermediate_images = []
        
        # Custom callback to capture intermediate states
        def callback_fn(step, timestep, latents):
            if request.return_intermediates and step % max(1, request.num_inference_steps // 10) == 0:
                # Store latent statistics
                latents_np = latents.cpu().numpy()
                intermediate_steps.append({
                    'step': step,
                    'timestep': float(timestep),
                    'latent_shape': list(latents.shape),
                    'latent_stats': {
                        'mean': float(latents_np.mean()),
                        'std': float(latents_np.std()),
                        'min': float(latents_np.min()),
                        'max': float(latents_np.max())
                    }
                })
                
                # Decode intermediate latent to image (optional, can be slow)
                try:
                    with torch.no_grad():
                        # VAE decode
                        latents_1_over_sigma = 1 / pipe.vae.config.scaling_factor
                        image = pipe.vae.decode(latents * latents_1_over_sigma).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = (image * 255).astype(np.uint8)
                        
                        # Convert to base64
                        img_pil = Image.fromarray(image[0])
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        intermediate_images.append({
                            'step': step,
                            'timestep': float(timestep),
                            'image': img_base64
                        })
                except Exception:
                    # Skip image decoding if it fails (can be memory intensive)
                    pass
        
        # Generate image
        with torch.no_grad():
            result = pipe(
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                callback=callback_fn if request.return_intermediates else None,
                callback_steps=1
            )
        
        # Convert final image to base64
        final_image = result.images[0]
        buffer = io.BytesIO()
        final_image.save(buffer, format='PNG')
        final_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Get model architecture info
        unet = pipe.unet
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        
        architecture_info = {
            'unet': {
                'input_channels': unet.config.in_channels,
                'output_channels': unet.config.out_channels,
                'block_out_channels': unet.config.block_out_channels,
                'attention_head_dim': unet.config.attention_head_dim,
                'num_attention_heads': unet.config.num_attention_heads,
                'cross_attention_dim': unet.config.cross_attention_dim,
            },
            'vae': {
                'latent_channels': vae.config.latent_channels,
                'sample_size': vae.config.sample_size,
            },
            'text_encoder': {
                'max_position_embeddings': text_encoder.config.max_position_embeddings,
                'hidden_size': text_encoder.config.hidden_size,
            },
            'scheduler': {
                'type': type(pipe.scheduler).__name__,
                'num_train_timesteps': pipe.scheduler.config.num_train_timesteps,
            }
        }
        
        return {
            "image": final_image_base64,
            "prompt": request.prompt,
            "intermediate_steps": intermediate_steps,
            "intermediate_images": intermediate_images,
            "architecture_info": architecture_info,
            "generation_params": {
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/diffusion/{model_id}/architecture")
async def get_diffusion_architecture(model_id: str):
    """Get architecture information for a diffusion model"""
    if model_id not in loaded_diffusion_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = loaded_diffusion_models[model_id]
    pipe = model_data['pipeline']
    
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    
    layers = []
    
    # UNet layers
    for name, module in unet.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_type = type(module).__name__
            layers.append({
                "name": name,
                "type": layer_type,
                "parameters": sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
            })
    
    return {
        "model_name": model_data['model_name'],
        "components": {
            "unet": {
                "type": "U-Net (Noise Predictor)",
                "description": "Predicts noise to remove at each denoising step",
                "input_channels": unet.config.in_channels,
                "output_channels": unet.config.out_channels,
                "layers": len([l for l in layers if 'unet' in l['name'].lower() or 'down' in l['name'].lower() or 'up' in l['name'].lower()])
            },
            "vae": {
                "type": "VAE (Variational Autoencoder)",
                "description": "Encodes images to latent space and decodes back",
                "latent_channels": vae.config.latent_channels,
            },
            "text_encoder": {
                "type": "CLIP Text Encoder",
                "description": "Encodes text prompt into embeddings",
                "hidden_size": text_encoder.config.hidden_size,
            },
            "scheduler": {
                "type": type(pipe.scheduler).__name__,
                "description": "Controls the denoising schedule",
            }
        },
        "total_params": sum(p.numel() for p in pipe.unet.parameters()) + 
                       sum(p.numel() for p in pipe.vae.parameters()) + 
                       sum(p.numel() for p in pipe.text_encoder.parameters())
    }

# ============== STATE SPACE MODELS ==============

# Storage for state space models
loaded_state_space_models = {}

# State space model configurations
STATE_SPACE_MODELS = {
    'mamba-130m': {
        'hf_name': 'state-spaces/mamba-130m',
        'description': 'Mamba 130M - Efficient state space model for sequences',
        'd_model': 768,
        'n_layers': 24
    },
    'mamba-370m': {
        'hf_name': 'state-spaces/mamba-370m',
        'description': 'Mamba 370M - Medium-sized state space model',
        'd_model': 1024,
        'n_layers': 48
    },
    'mamba-790m': {
        'hf_name': 'state-spaces/mamba-790m',
        'description': 'Mamba 790M - Large state space model',
        'd_model': 1536,
        'n_layers': 48
    },
}

class StateSpaceInferenceRequest(BaseModel):
    text: str
    max_length: int = 100
    return_states: bool = True  # Return intermediate states for visualization

@app.post("/state-space/load")
async def load_state_space_model(model_name: str):
    """Load a state space model (Mamba)"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        if model_name not in STATE_SPACE_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        config = STATE_SPACE_MODELS[model_name]
        hf_name = config['hf_name']
        
        model_id = f"{model_name}_{len(loaded_state_space_models)}"
        
        # Try to load from HuggingFace - many Mamba models are available there
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try loading the model
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model.eval()
            except Exception as model_error:
                # If model loading fails, provide helpful error message
                error_msg = str(model_error)
                if "mamba" in error_msg.lower() or "state" in error_msg.lower():
                    raise HTTPException(
                        status_code=500,
                        detail=f"State space model loading requires additional setup. "
                               f"Note: mamba-ssm package may require CUDA and complex build setup. "
                               f"Error: {error_msg}. "
                               f"For Mac/CPU, you may need to use alternative implementations or pre-built wheels."
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load model: {error_msg}"
                    )
            
            loaded_state_space_models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name,
                'config': config
            }
            
            return {
                "model_id": model_id,
                "model_name": model_name,
                "description": config['description'],
                "d_model": config['d_model'],
                "n_layers": config['n_layers']
            }
        except HTTPException:
            raise
        except Exception as e:
            # Provide helpful error message about installation
            error_msg = str(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load state space model. "
                       f"State space models (like Mamba) may require the mamba-ssm package, "
                       f"which has complex build requirements (CUDA, nvcc). "
                       f"For demonstration purposes, you can use HuggingFace transformers models. "
                       f"Error: {error_msg}"
            )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Required library not found: {str(e)}. Install with: pip install transformers torch"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/state-space/{model_id}/infer")
async def state_space_inference(model_id: str, request: StateSpaceInferenceRequest):
    """Run inference on a state space model with step-by-step visualization"""
    if model_id not in loaded_state_space_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import torch
        
        model_data = loaded_state_space_models[model_id]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        # Tokenize input
        inputs = tokenizer(request.text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=request.return_states)
        
        # Extract hidden states if available
        hidden_states = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                hidden_states.append({
                    "layer": layer_idx,
                    "shape": list(hidden_state.shape),
                    "mean": float(hidden_state.mean().item()),
                    "std": float(hidden_state.std().item()),
                    "min": float(hidden_state.min().item()),
                    "max": float(hidden_state.max().item()),
                    "sample": hidden_state[0, -1, :10].cpu().numpy().tolist()  # First 10 dims of last token
                })
        
        # Get logits for next token prediction
        logits = outputs.logits[0, -1, :]  # Last position, all vocab
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)
        
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode([idx.item()])
            top_predictions.append({
                "token": token,
                "probability": float(prob.item()),
                "logit": float(logits[idx].item())
            })
        
        # Get model architecture info
        architecture_info = {
            "d_model": model_data['config']['d_model'],
            "n_layers": model_data['config']['n_layers'],
            "vocab_size": len(tokenizer),
            "model_type": "state_space"
        }
        
        # State space specific info
        state_space_info = {
            "state_dim": model_data['config']['d_model'],
            "sequence_length": input_ids.shape[1],
            "efficient_attention": True,
            "linear_complexity": True
        }
        
        return {
            "input_tokens": tokens,
            "hidden_states": hidden_states,
            "top_predictions": top_predictions,
            "architecture_info": architecture_info,
            "state_space_info": state_space_info,
            "logits_stats": {
                "min": float(logits.min().item()),
                "max": float(logits.max().item()),
                "mean": float(logits.mean().item()),
                "std": float(logits.std().item())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/state-space/{model_id}/architecture")
async def get_state_space_architecture(model_id: str):
    """Get architecture information for a state space model"""
    if model_id not in loaded_state_space_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = loaded_state_space_models[model_id]
    model = model_data['model']
    config = model_data['config']
    
    # Try to extract layer information
    layers = []
    try:
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_type = type(module).__name__
                layers.append({
                    "name": name,
                    "type": layer_type,
                    "parameters": sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
                })
    except:
        pass
    
    return {
        "model_name": model_data['model_name'],
        "components": {
            "state_space_block": {
                "type": "Mamba Block",
                "description": "Core state space block with selective state spaces",
                "d_model": config['d_model'],
                "n_layers": config['n_layers']
            },
            "ssm": {
                "type": "State Space Model",
                "description": "Efficiently processes sequences using state space equations",
            },
            "tokenizer": {
                "type": "Tokenizer",
                "description": "Converts text to token IDs",
                "vocab_size": len(model_data['tokenizer'])
            }
        },
        "total_params": sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0,
        "layers": layers[:50]  # Limit to first 50 layers
    }

# ============== WORLD MODELS ==============

# Storage for world models
loaded_world_models = {}

# World model configurations
WORLD_MODELS = {
    'world-model-v1': {
        'description': 'World Model v1 - VAE encoder, RNN memory, MLP controller',
        'latent_dim': 32,
        'hidden_dim': 256,
        'action_dim': 3
    },
}

class WorldModelEncodeRequest(BaseModel):
    image: Optional[str] = None  # Base64 encoded image
    return_reconstruction: bool = True

class WorldModelPredictRequest(BaseModel):
    latent: Optional[list] = None  # Latent vector
    action: Optional[list] = None  # Action taken
    steps: int = 1  # Number of steps to predict ahead

class WorldModelControlRequest(BaseModel):
    latent: list  # Current latent state

class WorldModelSimulateRequest(BaseModel):
    initial_image: Optional[str] = None  # Base64 encoded initial observation
    steps: int = 10  # Number of simulation steps
    actions: Optional[list] = None  # Optional: provide actions, otherwise use controller

@app.post("/world-models/load")
async def load_world_model(model_name: str = "world-model-v1"):
    """Load a world model"""
    try:
        import torch
        import torch.nn as nn
        
        if model_name not in WORLD_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        config = WORLD_MODELS[model_name]
        model_id = f"{model_name}_{len(loaded_world_models)}"
        
        # Create a simplified World Model implementation for demonstration
        # In practice, you would load a pre-trained model
        
        class VisionEncoder(nn.Module):
            """V: Vision encoder - compresses observations to latent space"""
            def __init__(self, latent_dim=32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 4, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, latent_dim * 2)  # Mean and logvar
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Unflatten(1, (256, 1, 1)),
                    nn.ConvTranspose2d(256, 128, 4, 1, 0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),
                    nn.Sigmoid()
                )
            
            def encode(self, x):
                h = self.encoder(x)
                mu, logvar = h.chunk(2, dim=1)
                return mu, logvar
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
        
        class MemoryRNN(nn.Module):
            """M: Memory - RNN that predicts future latent states"""
            def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
                self.predictor = nn.Linear(hidden_dim, latent_dim)
            
            def forward(self, latent, action, hidden=None):
                # Concatenate latent and action
                input_vec = torch.cat([latent, action], dim=-1)
                # Add sequence dimension if needed
                if input_vec.dim() == 2:
                    input_vec = input_vec.unsqueeze(1)
                lstm_out, hidden = self.lstm(input_vec, hidden)
                predicted_latent = self.predictor(lstm_out)
                return predicted_latent, hidden
        
        class Controller(nn.Module):
            """C: Controller - policy network that generates actions from latent states"""
            def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
                super().__init__()
                self.policy = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Tanh()  # Actions typically in [-1, 1]
                )
            
            def forward(self, latent):
                return self.policy(latent)
        
        # Create model components
        vision = VisionEncoder(config['latent_dim'])
        memory = MemoryRNN(config['latent_dim'], config['hidden_dim'], config['action_dim'])
        controller = Controller(config['latent_dim'], config['hidden_dim'], config['action_dim'])
        
        # Initialize with better weights for demonstration
        # Use smaller initialization for more stable outputs
        for module in [vision, memory]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)  # Smaller gain for more stable outputs
                else:
                    nn.init.constant_(param, 0.0)  # Initialize biases to zero
        
        # Initialize controller with larger weights to produce more varied actions
        for name, param in controller.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=1.0)  # Larger gain for more action variation
            else:
                nn.init.constant_(param, 0.0)  # Initialize biases to zero
        
        # Initialize decoder with better weights to produce more varied outputs
        # This makes the visualization more interesting even without full training
        with torch.no_grad():
            for name, param in vision.decoder.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    # Use normal initialization with some variation
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    # Small positive bias for decoder to encourage non-zero outputs
                    nn.init.constant_(param, 0.01)
        
        vision.eval()
        memory.eval()
        controller.eval()
        
        loaded_world_models[model_id] = {
            'vision': vision,
            'memory': memory,
            'controller': controller,
            'model_name': model_name,
            'config': config
        }
        
        return {
            "model_id": model_id,
            "model_name": model_name,
            "description": config['description'],
            "latent_dim": config['latent_dim'],
            "hidden_dim": config['hidden_dim'],
            "action_dim": config['action_dim']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load world model: {str(e)}")

@app.post("/world-models/{model_id}/encode")
async def world_model_encode(model_id: str, request: WorldModelEncodeRequest):
    """Encode observation through V (Vision) component"""
    if model_id not in loaded_world_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import torch
        import base64
        
        model_data = loaded_world_models[model_id]
        vision = model_data['vision']
        
        # If image provided, decode and process it
        if request.image:
            image_data = base64.b64decode(request.image.split(',')[-1])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = image.resize((64, 64))
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0)
        else:
            # Generate random image for demonstration
            image_tensor = torch.rand(1, 3, 64, 64)
        
        with torch.no_grad():
            mu, logvar = vision.encode(image_tensor)
            latent = vision.reparameterize(mu, logvar)
            
            reconstruction = None
            if request.return_reconstruction:
                reconstruction = vision.decode(latent)
                # Normalize reconstruction to [0, 1] range for better visualization
                # Use percentile-based normalization to handle outliers better
                recon_tensor = reconstruction[0]
                recon_flat = recon_tensor.flatten()
                recon_p5 = torch.quantile(recon_flat, 0.05)
                recon_p95 = torch.quantile(recon_flat, 0.95)
                
                if recon_p95 > recon_p5:
                    # Normalize using 5th and 95th percentiles to avoid extreme outliers
                    recon_tensor = (recon_tensor - recon_p5) / (recon_p95 - recon_p5)
                    recon_tensor = recon_tensor.clamp(0, 1)
                else:
                    # If all values are similar, add some contrast
                    recon_mean = recon_tensor.mean()
                    recon_tensor = (recon_tensor - recon_mean) * 2.0 + 0.5
                    recon_tensor = recon_tensor.clamp(0, 1)
                
                # Convert to base64
                recon_img = transforms.ToPILImage()(recon_tensor)
                buffer = io.BytesIO()
                recon_img.save(buffer, format='PNG')
                reconstruction_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                reconstruction_b64 = None
        
        return {
            "latent": latent[0].cpu().numpy().tolist(),
            "mu": mu[0].cpu().numpy().tolist(),
            "logvar": logvar[0].cpu().numpy().tolist(),
            "reconstruction": reconstruction_b64,
            "latent_stats": {
                "mean": float(latent.mean().item()),
                "std": float(latent.std().item()),
                "min": float(latent.min().item()),
                "max": float(latent.max().item())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

@app.post("/world-models/{model_id}/predict")
async def world_model_predict(model_id: str, request: WorldModelPredictRequest):
    """Predict next state through M (Memory) component"""
    if model_id not in loaded_world_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import torch
        
        model_data = loaded_world_models[model_id]
        memory = model_data['memory']
        latent_dim = model_data['config']['latent_dim']
        action_dim = model_data['config']['action_dim']
        
        # Prepare inputs
        if request.latent:
            latent = torch.tensor(request.latent, dtype=torch.float32).unsqueeze(0)
        else:
            latent = torch.randn(1, latent_dim)
        
        if request.action:
            action = torch.tensor(request.action, dtype=torch.float32).unsqueeze(0)
        else:
            action = torch.randn(1, action_dim)
        
        predictions = []
        hidden = None
        
        with torch.no_grad():
            for step in range(request.steps):
                predicted_latent, hidden = memory(latent, action, hidden)
                predictions.append({
                    "step": step + 1,
                    "predicted_latent": predicted_latent[0, 0].cpu().numpy().tolist(),
                    "stats": {
                        "mean": float(predicted_latent.mean().item()),
                        "std": float(predicted_latent.std().item()),
                        "min": float(predicted_latent.min().item()),
                        "max": float(predicted_latent.max().item())
                    }
                })
                # Use predicted latent for next step
                latent = predicted_latent[0, 0].unsqueeze(0)
        
        return {
            "predictions": predictions,
            "initial_latent": latent[0].cpu().numpy().tolist() if request.latent else None,
            "action_used": action[0].cpu().numpy().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/world-models/{model_id}/control")
async def world_model_control(model_id: str, request: WorldModelControlRequest):
    """Generate action through C (Controller) component"""
    if model_id not in loaded_world_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import torch
        
        model_data = loaded_world_models[model_id]
        controller = model_data['controller']
        
        latent = torch.tensor(request.latent, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action = controller(latent)
            # Get action probabilities (softmax over action space)
            action_probs = torch.softmax(action, dim=-1)
        
        return {
            "action": action[0].cpu().numpy().tolist(),
            "action_probabilities": action_probs[0].cpu().numpy().tolist(),
            "action_stats": {
                "mean": float(action.mean().item()),
                "std": float(action.std().item()),
                "min": float(action.min().item()),
                "max": float(action.max().item())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Control failed: {str(e)}")

@app.post("/world-models/{model_id}/simulate")
async def world_model_simulate(model_id: str, request: WorldModelSimulateRequest):
    """Run full world model simulation showing all components"""
    if model_id not in loaded_world_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import torch
        import base64
        
        model_data = loaded_world_models[model_id]
        vision = model_data['vision']
        memory = model_data['memory']
        controller = model_data['controller']
        
        steps = []
        hidden = None
        
        # Initial observation
        if request.initial_image:
            image_data = base64.b64decode(request.initial_image.split(',')[-1])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = image.resize((64, 64))
            transform = transforms.Compose([transforms.ToTensor()])
            current_obs = transform(image).unsqueeze(0)
        else:
            current_obs = torch.rand(1, 3, 64, 64)
        
        with torch.no_grad():
            for step in range(request.steps):
                # Store the current observation (input to this step) for visualization
                # For step 0, this is the initial observation
                # For step 1+, this is the predicted observation from the previous step
                # Normalize it for consistent visualization
                obs_tensor = current_obs[0]
                obs_flat = obs_tensor.flatten()
                obs_p5 = torch.quantile(obs_flat, 0.05)
                obs_p95 = torch.quantile(obs_flat, 0.95)
                
                if obs_p95 > obs_p5:
                    obs_tensor_normalized = (obs_tensor - obs_p5) / (obs_p95 - obs_p5)
                    obs_tensor_normalized = obs_tensor_normalized.clamp(0, 1)
                else:
                    obs_mean = obs_tensor.mean()
                    obs_tensor_normalized = (obs_tensor - obs_mean) * 2.0 + 0.5
                    obs_tensor_normalized = obs_tensor_normalized.clamp(0, 1)
                
                obs_img = transforms.ToPILImage()(obs_tensor_normalized)
                obs_buffer = io.BytesIO()
                obs_img.save(obs_buffer, format='PNG')
                current_obs_b64 = base64.b64encode(obs_buffer.getvalue()).decode('utf-8')
                
                # V: Encode observation
                mu, logvar = vision.encode(current_obs)
                latent = vision.reparameterize(mu, logvar)
                
                # C: Generate action
                if request.actions and step < len(request.actions):
                    action = torch.tensor(request.actions[step], dtype=torch.float32).unsqueeze(0)
                else:
                    action = controller(latent)
                
                # M: Predict next latent
                predicted_latent, hidden = memory(latent, action, hidden)
                
                # Decode predicted latent to visualize
                # predicted_latent shape: [batch, seq, latent_dim] = [1, 1, 32]
                # We need [batch, latent_dim] = [1, 32] for decoder
                predicted_latent_for_decode = predicted_latent[0, 0].unsqueeze(0)  # [1, 32]
                predicted_obs = vision.decode(predicted_latent_for_decode)  # [1, 3, 64, 64]
                
                # Normalize predicted image to [0, 1] range for better visualization
                # Use percentile-based normalization to handle outliers better
                pred_tensor = predicted_obs[0]
                pred_flat = pred_tensor.flatten()
                pred_p5 = torch.quantile(pred_flat, 0.05)
                pred_p95 = torch.quantile(pred_flat, 0.95)
                
                if pred_p95 > pred_p5:
                    # Normalize using 5th and 95th percentiles to avoid extreme outliers
                    pred_tensor_normalized = (pred_tensor - pred_p5) / (pred_p95 - pred_p5)
                    pred_tensor_normalized = pred_tensor_normalized.clamp(0, 1)
                else:
                    # If all values are similar, add some contrast
                    pred_mean = pred_tensor.mean()
                    pred_tensor_normalized = (pred_tensor - pred_mean) * 2.0 + 0.5
                    pred_tensor_normalized = pred_tensor_normalized.clamp(0, 1)
                
                pred_img = transforms.ToPILImage()(pred_tensor_normalized)
                pred_buffer = io.BytesIO()
                pred_img.save(pred_buffer, format='PNG')
                
                steps.append({
                    "step": step + 1,
                    "observation": current_obs_b64,  # This is the input to this step (normalized for display)
                    "latent": latent[0].cpu().numpy().tolist(),
                    "action": action[0].cpu().numpy().tolist(),
                    "predicted_latent": predicted_latent[0, 0].cpu().numpy().tolist(),
                    "predicted_observation": base64.b64encode(pred_buffer.getvalue()).decode('utf-8')
                })
                
                # Use predicted observation for next step (dreaming)
                # Use the raw predicted_obs (not normalized) for the next encoding step
                # The normalization is only for visualization
                current_obs = predicted_obs  # Keep as [1, 3, 64, 64] for next iteration
        
        return {
            "steps": steps,
            "total_steps": request.steps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/world-models/{model_id}/architecture")
async def get_world_model_architecture(model_id: str):
    """Get architecture information for a world model"""
    if model_id not in loaded_world_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = loaded_world_models[model_id]
    config = model_data['config']
    vision = model_data['vision']
    memory = model_data['memory']
    controller = model_data['controller']
    
    return {
        "model_name": model_data['model_name'],
        "components": {
            "vision": {
                "type": "VAE Encoder-Decoder",
                "description": "V: Compresses observations to latent space and can reconstruct them",
                "latent_dim": config['latent_dim'],
                "encoder_params": sum(p.numel() for p in vision.encoder.parameters()),
                "decoder_params": sum(p.numel() for p in vision.decoder.parameters())
            },
            "memory": {
                "type": "LSTM RNN",
                "description": "M: Predicts future latent states given current state and action",
                "hidden_dim": config['hidden_dim'],
                "params": sum(p.numel() for p in memory.parameters())
            },
            "controller": {
                "type": "MLP Policy Network",
                "description": "C: Generates actions from latent states",
                "action_dim": config['action_dim'],
                "params": sum(p.numel() for p in controller.parameters())
            }
        },
        "total_params": sum(p.numel() for p in vision.parameters()) + 
                       sum(p.numel() for p in memory.parameters()) + 
                       sum(p.numel() for p in controller.parameters())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

