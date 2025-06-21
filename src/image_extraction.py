from PIL import Image
import cv2
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import layoutparser as lp
from typing import List, Tuple, Optional, Union, Dict, Any
import os
import logging

class ImageExtraction: 
  
  def __init__(self, model_path: str, label_dict: Dict[int, str], threshold: float) -> None:
    """Initialize the image extraction model.
    
    Args:
        model_path: Path to the layout detection model
        label_dict: Dictionary mapping label IDs to names
        threshold: Detection confidence threshold
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model initialization fails
    """
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model file not found: {model_path}")
    
    self.model_path = model_path 
    self.label_dict = label_dict
    self.threshold = threshold 
    
    try:
      self.ocr_agent = lp.TesseractAgent(languages='eng')
      self.model = lp.Detectron2LayoutModel(
        self.model_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.threshold],
        label_map=self.label_dict
      )
    except Exception as e:
      logging.error(f"Failed to initialize model: {str(e)}")
      raise Exception(f"Model initialization failed: {str(e)}")
    
    # Initialize instance variables
    self.layout: Optional[lp.Layout] = None
    self.page: Optional[Image.Image] = None
    self.text_blocks: Optional[List[Any]] = None 
    self.figure_blocks: Optional[List[Any]] = None
    self.title_blocks: Optional[List[Any]] = None
    
  def get_layout(self, image_path: str) -> None:
    """Get a layout object for a specified image.
    
    Args:
        image_path: Path to the image file
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If layout detection fails
    """
    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
      self.page = Image.open(image_path)
      self.layout = self.model.detect(self.page)
    except Exception as e:
      logging.error(f"Failed to detect layout for {image_path}: {str(e)}")
      raise Exception(f"Layout detection failed: {str(e)}")

  def get_image(self, textblock: Any) -> np.ndarray:
    """Returns the image array for a specific section of an image.
    
    Args:
        textblock: Layout block to extract image from
        
    Returns:
        Numpy array of the cropped image
        
    Raises:
        ValueError: If page is not loaded
        Exception: If image cropping fails
    """
    if self.page is None:
      raise ValueError("No page loaded. Call get_layout() first.")
    
    try:
      return np.asarray(textblock.crop_image(np.array(self.page)))
    except Exception as e:
      logging.error(f"Failed to crop image: {str(e)}")
      raise Exception(f"Image cropping failed: {str(e)}")

  def return_blocks(self) -> None:
    """Categorize layout blocks by type.
    
    Raises:
        ValueError: If layout is not loaded
    """
    if self.layout is None:
      raise ValueError("No layout loaded. Call get_layout() first.")
    
    try:
      text_blocks = []
      figure_blocks = []
      title_blocks = []

      for block in self.layout:
        if block.type == 'Figure':
          figure_blocks.append(block)
        elif block.type == 'Text':
          text_blocks.append(block)
        elif block.type == 'Title':
          title_blocks.append(block)

      self.text_blocks = text_blocks
      self.figure_blocks = figure_blocks
      self.title_blocks = title_blocks
    except Exception as e:
      logging.error(f"Failed to categorize blocks: {str(e)}")
      raise Exception(f"Block categorization failed: {str(e)}")

  def identify_title(self, coordinate_of_figure: Tuple[float, float, float, float], 
                    x_tolerance: float = 0.2, y_tolerance: float = 0.2) -> Union[Any, str]:
    """Provides the title for a given figure based on coordinate proximity.
    
    Args:
        coordinate_of_figure: Tuple of (x1, y1, x2, y2) coordinates
        x_tolerance: Horizontal tolerance for matching (default 0.2 = 20%)
        y_tolerance: Vertical tolerance for matching (default 0.2 = 20%)
        
    Returns:
        Title block if found, otherwise "No title block found"
        
    Raises:
        ValueError: If title_blocks is not initialized
    """
    if self.title_blocks is None:
      raise ValueError("Title blocks not initialized. Call return_blocks() first.")
    
    try:
      for block in self.title_blocks:
        x_match = (coordinate_of_figure[0] * (1 - x_tolerance) <= block.coordinates[0] <= 
                  coordinate_of_figure[0] * (1 + x_tolerance))
        y_match = (coordinate_of_figure[3] <= block.coordinates[1] <= 
                  coordinate_of_figure[3] * (1 + y_tolerance))
        
        if x_match and y_match:
          return block
      
      return "No title block found"
    except Exception as e:
      logging.error(f"Failed to identify title: {str(e)}")
      raise Exception(f"Title identification failed: {str(e)}") 


  def get_text_and_image(self) -> List[Tuple[str, np.ndarray]]:
    """Extract text and image pairs from figures and their titles.
    
    Returns:
        List of tuples containing (text, image_array) pairs
        
    Raises:
        ValueError: If required blocks are not initialized
        Exception: If text/image extraction fails
    """
    if self.figure_blocks is None:
      raise ValueError("Figure blocks not initialized. Call return_blocks() first.")
    
    try:
      figures_title = []
      text_and_image = []

      # Identify title blocks for each figure
      for figure in self.figure_blocks: 
        title_block = self.identify_title(figure.coordinates)
        figures_title.append([figure, title_block])

      # Extract text and images
      for figure, title in figures_title:
        if title != "No title block found":
          try:
            # Pad and crop the title segment
            segment_image = (title
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(np.array(self.page)))
            
            # Extract text using OCR
            text = self.ocr_agent.detect(segment_image)
            
            # Get figure image
            figure_image = self.get_image(figure)
            
            text_and_image.append((text, figure_image))
          except Exception as e:
            logging.warning(f"Failed to process figure-title pair: {str(e)}")
            continue
        else:
          logging.warning("Figure found without corresponding title")
      
      return text_and_image
    except Exception as e:
      logging.error(f"Failed to extract text and images: {str(e)}")
      raise Exception(f"Text and image extraction failed: {str(e)}")
