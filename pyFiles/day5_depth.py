import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import requests
import matplotlib.pyplot as plt

model_id = "Intel/dpt-large"
processor = DPTImageProcessor.from_pretrained(model_id)
model = DPTForDepthEstimation.from_pretrained(model_id)

url = "../images/cup_1.jpeg"
image = Image.open(url)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)


output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / output.max()).astype("uint8")
depth_image = Image.fromarray(formatted)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(depth_image, cmap='inferno') # 'inferno' or 'magma' make depth look cool
ax[1].set_title("Predicted Depth Map (MiDaS)")
ax[1].axis("off")
plt.show()