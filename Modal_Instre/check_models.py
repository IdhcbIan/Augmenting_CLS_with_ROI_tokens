import timm

# List all available models with "dino" in their name
dino_models = [model for model in timm.list_models() if "dino" in model.lower()]
print("Available DINO models in timm:")
for model in dino_models:
    print(f"- {model}")

# Try to print details about a specific model if available
try:
    if dino_models:
        model_name = dino_models[0]
        print(f"\nDetails for {model_name}:")
        model = timm.create_model(model_name, pretrained=False)
        print(f"  - Features dimension: {model.num_features}")
        print(f"  - Default image size: {model.default_cfg.get('input_size', 'Unknown')}")
except Exception as e:
    print(f"Error getting model details: {e}")

print("\nTimm version:", timm.__version__)
