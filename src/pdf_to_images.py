from pdf2image import convert_from_path

class pdf_to_images: 
  # convert pdf to individual images
  def __init__(self, pdf_path): 
    self.images = convert_from_path(pdf_path)
    self.image_list = list()

    for i in range(len(self.images)):
      # Save pages as images in the pdf
      self.image_list.append('page'+ str(i) +'.jpg')
      self.images[i].save('page'+ str(i) +'.jpg', 'JPEG')