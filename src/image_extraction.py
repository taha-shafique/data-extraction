try:
    from PIL import Image
except:
    import Image
import cv2
import pandas  as pd
import numpy as np 
import matplotlib.pyplot as plt
import layoutparser as lp


class image_extraction: 
  
  def __init__(self, model_path, label_dict, threshold): 
    self.model_path = model_path 
    self.label_dict= label_dict
    self.threshold = threshold 
    self.ocr_agent = lp.TesseractAgent(languages='eng') ####
    self.layout = None # layout is none when its instantiated. But will be changed when get_layout 
      # method is called. 
    self.page = None 

    self.text_blocks = None 
    self.figure_blocks = None
    self.title_blocks = None 


    self.model = lp.Detectron2LayoutModel(self.model_path,
                                     extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.threshold],
                                     label_map = self.label_dict)

    #class variables:

    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
    
  def get_layout(self, image): 
    '''Get a layout object for a specified image'''
    self.layout = self.model.detect(Image.open(image)) # returns a layoutparser object 
    self.page = Image.open(image)


  def get_image(self, textblock): 
    '''returns the image inline for a specfic section of an image'''
    #return plt.imshow(self.layout[position].crop_image(np.array(self.page)))
    return np.asarray(textblock.crop_image(np.array(self.page)))
    

  def return_blocks(self): 
    # This needs re-work. How would this be scalable if other object types are used in label_map?

    text_blocks = []
    figure_blocks = []
    title_blocks = []

    for i in range(len(self.layout)):

      if self.layout[i].type == 'Figure': #if layout type is figure, append to the figure block list 
        figure_blocks.append(self.layout[i])

      elif self.layout[i].type == 'Text': #if layout type is figure, append to the figure block list 
        text_blocks.append(self.layout[i])

      elif self.layout[i].type == 'Title': #if layout type is figure, append to the figure block list 
        title_blocks.append(self.layout[i])
      #figure_blocks = lp.Layout([b for b in layout if b.type=='Figure']


    self.text_blocks = text_blocks
    self.figure_blocks = figure_blocks
    self.title_blocks = title_blocks

    # return text_blocks, figure_blocks, title_blocks


# where to get list to title blocks, and coordinate of figure

  def identify_title(self, coordinate_of_figure): 
    '''Provides the title for a given picture'''
    counter = 0 
    for block in self.title_blocks: #This returns the TextBlock in the list
      counter += 1 
      argument_1 = block.coordinates[0] >= coordinate_of_figure[0]*.80 and block.coordinates[0] <= coordinate_of_figure[0]*1.20
      #argument_2 = block.coordinates[2] >= coordinate_of_figure[2]*.95 and block.coordinates[2] <= coordinate_of_figure[2]*1.05
      argument_3 = block.coordinates[1] >= coordinate_of_figure[3] and block.coordinates[1] <= coordinate_of_figure[3]*1.20
      
      if argument_1 and argument_3: 
        return block

      elif counter == len(self.title_blocks): 
        return "No title block found"

      else: 
        continue 


  def get_text_and_image(self): 

    figures_title = []
    text_and_image = []

    # first for loop identifies the blocks
    for figure in self.figure_blocks: 
      figures_title.append([figure, self.identify_title(figure.coordinates)])

    # second for loop identifies the text in the blocks
    for i in range(len(figures_title)):
      

      segment_image = (figures_title[i][1]
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(np.array(self.page)))
    

      text = self.ocr_agent.detect(segment_image)
      
      text_and_image.append([text, self.get_image(figures_title[i][0])])
    
    return text_and_image
