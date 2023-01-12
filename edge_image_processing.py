#!/usr/bin/env python3
import cv2
import numpy as np

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
    
def main():
    filename = 'frame-4500.tif'
    raw2d = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img8 = (raw2d).astype('uint8')
    
    # Get image with ROIs mapped out as nonzero values
    edge_detected, th = otsu_edge(img8)
    
    # This contours variable, you will find very usefull. 
    # It contains every outline of each cell!
    # It is a tuple containing numpy arrays
    # Each numpy array in the tuple contains a list of x-y points that 
    # Make up the boundaries of each cell
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours over existing raw image
    # You could draw this on any image you like
    cv2.drawContours(raw2d, contours, -1, (255,255,255), 1)
    
    
    # Display
    cv2.imshow('Raw image w/ Contours Overlayed', raw2d)
    cv2.waitKey(0)
    cv2.imshow('edge detection', edge_detected)
    cv2.waitKey(0)

if __name__=='__main__':
    main()

