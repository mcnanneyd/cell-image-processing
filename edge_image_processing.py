#!/usr/bin/env python3
import cv2
import numpy as np
import os
from typing import List
import re

def sort_by_number( string_list: List[str] ) -> List[str]: 
    """ Sort the """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(string_list, key = alphanum_key)

def otsu_edge(image, sigma=0.33):
    # Perform Gaussian blur to reduce noise, you may not need this so much
    blur = cv2.GaussianBlur(image,(5,5),0)
    
    # Create binary image of file (every pixel is either black or white)
    # This is stored in th, We are not using the other variable, so the 
    # convention is to unpack it with _
    _,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # compute the median of the single channel pixel intensities
    v = np.median(th)
    
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(th, lower, upper, apertureSize=3)
    
    # Edge detected image and threshold array
    return edged, th


def get_rois(   brightfield_image: str, lower_a: float = 50, 
                upper_a: float = 225):
    """Get ROIs filtered by roi area

    Args:
        brightfield_image (str): filename of brighfield image
        lower_a (float, optional): Lower bound for cell area. Defaults to 50.
        upper_a (float, optional): Upper bound for cell area. Defaults to 225.

    Returns:
        cells (tuple of numpy arrays containing points that define 
        the boundaries of each ROI): This is in the exact format that 
        opencv returns contours 
    """
    
    raw2d = cv2.imread(brightfield_image, cv2.IMREAD_GRAYSCALE)
    img8 = (raw2d).astype('uint8')
    
    # Get image with ROIs mapped out as nonzero values
    edge_detected, th = otsu_edge(img8)
    
    # This contours variable, you will find very usefull. 
    # It contains every outline of each cell!
    # It is a tuple containing numpy arrays
    # Each numpy array in the tuple contains a list of x-y points that 
    # Make up the boundaries of each cell
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    larger_contours = []
    
    # Filter rois by their pixel-area
    for cnt in contours:        
        area = cv2.contourArea(cnt)
        if area < lower_a or area > upper_a:
            continue
        larger_contours.append(cnt)
        
    cells = tuple(larger_contours)
    print(f"Roi detection complete - found {len(cells)} cells")

    return cells
    
def get_intensities(rois: tuple[np.array], image: np.array):
    intensities = np.zeros(len(rois), dtype=np.float64)
    # Allocate empty mask
    mask = np.zeros(image.shape, np.uint8)
    for i in range(len(rois)):
        # Clear the mask
        mask[...] = 0
        
        # Draw a single contour (index i) on empty mask
        cv2.drawContours(mask, rois, i, 255, -1)
        
        # Get average intensity in the specified roi
        intensity = cv2.mean(image, mask)[0]
        
        # Update value
        intensities[i] = intensity
        
    return intensities

def get_relative_intensities(rois: tuple[np.array], image_dir: str = '.'):
    files = sort_by_number([f for f in os.listdir(image_dir) \
        if (os.path.isfile(f) and f.endswith('.tif'))])

    first_image = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    print("first image", first_image)
    initial_intensities = get_intensities(rois, first_image)
    print(initial_intensities.shape)
    
    
    relative_intensities = np.zeros((len(files), initial_intensities.shape[0]))
    relative_intensities[...] = 1
    for i, file in enumerate(files[1:]):
        raw2d = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        intensities = get_intensities(rois, raw2d)
        relative_intensities[i+1] = intensities / initial_intensities
        
    return files, relative_intensities

def main():
    brightfield_image = '../StimSplit_01.tif'
    rois = get_rois(brightfield_image, 120, 125)
    filenames, intensities = get_relative_intensities(rois)
    print(intensities)
    
    
    
    
    

if __name__=='__main__':
    main()

