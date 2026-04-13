import numpy as np
import cv2
import supervision as sv


class Visualizer:
    """Visualizer for segmentation results"""
    
    def __init__(self, class_names, opacity=0.5):
        """
        Initialize visualizer
        
        Args:
            class_names: List of class names
            opacity: Mask overlay opacity (0.0 - 1.0)
        """
        self.class_names = class_names
        self.opacity = opacity
        
        # Define custom colors for specific classes
        self.class_colors = {
            1: (0, 255, 0),    # powerline -> Green
            2: (0, 0, 255),    # tower -> Red
        }
        
        # Create color palette
        self.color_palette = self._create_color_palette()
        
        # Initialize supervision annotators
        self.mask_annotator = sv.MaskAnnotator(
            color=self.color_palette,
            opacity=self.opacity
        )
        
        
    
    def _create_color_palette(self):
        """
        Create custom color palette for classes
        
        Returns:
            sv.ColorPalette: Color palette
        """
        num_classes = len(self.class_names)
        colors = [sv.Color(r=128, g=128, b=128)] * num_classes  # Default gray
        
        # Apply custom colors
        for class_id, (r, g, b) in self.class_colors.items():
            if class_id < num_classes:
                colors[class_id] = sv.Color(r=r, g=g, b=b)
        
        return sv.ColorPalette(colors=colors)
    
    def draw(self, image, detections):
        """
        Draw detections on image
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            detections: sv.Detections object
            
        Returns:
            numpy array: Annotated image
        """
        if len(detections) == 0:
            return image
        
        annotated_image = image.copy()
        
        # Draw masks
        if detections.mask is not None:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image,
                detections=detections
            )
        
        
        
        
        return annotated_image
    
    def draw_info(self, image, text, position=(10, 30), color=(0, 255, 0), thickness=2):
        """
        Draw text information on image
        
        Args:
            image: numpy array
            text: Text to draw
            position: (x, y) position
            color: BGR color tuple
            thickness: Text thickness
            
        Returns:
            numpy array: Image with text
        """
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness
        )
        return image

