import torch
from feature_extraction import SuperPointNet

# Load or define your model
model = SuperPointNet()
model.load_state_dict(torch.load("./superpoint_v1.pth"))
model.eval()

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)  # Use torch.jit.trace for purely sequential models

# Save the model
scripted_model.save("./superpoint_converted.pt")
