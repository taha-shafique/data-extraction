from pdf2image import convert_from_path
from typing import List, Optional
import os
import logging

class PDFToImages: 
  def __init__(self, pdf_path: str) -> None:
    """Convert PDF to individual images.
    
    Args:
        pdf_path: Path to the PDF file
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF conversion fails
    """
    if not os.path.exists(pdf_path):
      raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
      self.images = convert_from_path(pdf_path)
      self.image_list: List[str] = []
      self._convert_and_save_images()
    except Exception as e:
      logging.error(f"Failed to convert PDF {pdf_path}: {str(e)}")
      raise Exception(f"PDF conversion failed: {str(e)}")

  def _convert_and_save_images(self) -> None:
    """Convert and save PDF pages as JPEG images."""
    for i, image in enumerate(self.images):
      try:
        filename = f'page{i}.jpg'
        self.image_list.append(filename)
        image.save(filename, 'JPEG')
      except Exception as e:
        logging.error(f"Failed to save page {i}: {str(e)}")
        raise Exception(f"Failed to save page {i}: {str(e)}")
  
  def get_image_paths(self) -> List[str]:
    """Return list of saved image file paths."""
    return self.image_list.copy()