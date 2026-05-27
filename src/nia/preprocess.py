"""Preproces images.

General image transformations.
https://monai.readthedocs.io/en/stable/transforms.html

"""

from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    AddChannel
)

# Define transforms for image preprocessing
# Transforms. Domain-specific medical image transformations
transforms = Compose([
    LoadImage(image_only=True),
    AddChannel(),
    ScaleIntensity()
])

# Apply transforms to your image
image = transforms(image_path)