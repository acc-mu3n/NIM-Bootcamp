from PIL import Image
import io
import base64
import os

def process_image(image):
    """Resize image, encode as jpeg to shrink size, convert to b64 for upload, and save the resized image."""
    
    # Initialize file_path for the case when image is a filepath
    file_path = None
    
    # Handle different input types (filepath string or PIL Image)
    if isinstance(image, str):
        file_path = image
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    
    # Resize the image
    image = image.resize((336, 336))
    
    # Save the resized image with a "_resized" suffix if file_path is available
    if file_path:
        base, ext = os.path.splitext(file_path)
        resized_path = f"{base}_resized.jpg"
        image.save(resized_path, "JPEG")
    else:
        resized_path = None  # In cases where image is provided directly without a path

    # Encode as JPEG in-memory for uploading
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_data = buf.getvalue()
    
    # Convert to base64 string
    image_b64 = base64.b64encode(image_data).decode()
    
    # Ensure image is within acceptable size limits
    assert len(image_b64) < 180_000, "Image too large to upload."
    
    return image_b64