import torch
import cv2
import numpy as np
from transformers import pipeline

class DepthEstimator:
    def __init__(self):
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load MiDaS model for depth estimation
        self.pipe = pipeline(
            task="depth-estimation",
            model="Intel/dpt-large",
            device=0 if self.device == "cuda" else -1
        )
    
    def estimate_depth(self, image):
        """
        Estimate depth from RGB image
        Args:
            image: RGB numpy array
        Returns:
            depth_map: Normalized depth map
        """
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Get depth estimation
            result = self.pipe(pil_image)
            depth = result["depth"]
            
            # Convert to numpy array
            depth_array = np.array(depth)
            
            # Normalize depth values
            depth_normalized = cv2.normalize(
                depth_array, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            return depth_normalized
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            # Fallback: simple gradient-based depth
            return self._simple_depth_estimation(image)
    
    def _simple_depth_estimation(self, image):
        """Fallback depth estimation using image gradients"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Invert and normalize
        depth = 255 - cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        return depth