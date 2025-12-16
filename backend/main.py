from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor

app = FastAPI(title="Trinetr CNN Visualization API")

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
    return {"message": "Trinetr CNN Visualization API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/models/load")
async def load_model(model_name: str = "resnet18"):
    """Load a pre-trained CNN model"""
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
                layer_info["description"] = f"Fully connected layer - {module.out_features} outputs"
            else:
                layer_info["description"] = f"{layer_type} layer"
            
            layers.append(layer_info)
            layer_map[name] = layer_info
    
    # Build connections (simplified - sequential for most CNNs)
    for i in range(len(layers) - 1):
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
                "mean": float(activations.mean())
            }
        elif len(activations.shape) == 1:  # Linear layer
            return {
                "shape": list(activations.shape),
                "activations": activations.tolist(),
                "min": float(activations.min()),
                "max": float(activations.max()),
                "mean": float(activations.mean())
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

