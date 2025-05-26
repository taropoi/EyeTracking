import json
import yaml

# 从文件读取 JSON
training_params = json.load(
    open("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v4\\output\\wandb\\train-1\\training_params.json", "r")
)
dataset_params = json.load(
    open("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v4\\output\\wandb\\train-1\\dataset_params.json", "r")
)
layers_config = json.load(
    open("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v4\\output\\wandb\\train-1\\layer_configs.json", "r")
)

# 转成 YAML
yaml.dump(training_params, open("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v4\\output\\wandb\\train-1\\training_params.yaml", "w"))
yaml.dump(dataset_params, open("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v4\\output\\wandb\\train-1\\dataset_params.yaml", "w"))
yaml.dump(layers_config, open("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v4\\output\\wandb\\train-1\\layer_configs.yaml", "w"))

