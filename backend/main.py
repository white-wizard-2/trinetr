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

# Interpolation methods mapping
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

