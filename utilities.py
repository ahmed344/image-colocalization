import numpy as np
from skimage.filters import threshold_otsu
from scipy import linalg

def background_correction(img):
    """Background correction"""
    # Write the data in terms of 3-dim points excluding the contact zone
    coord_background_intensity = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            coord_background_intensity.append([i, j, img[i,j]])

    # 3-dim data points
    data = np.array(coord_background_intensity)

    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = linalg.lstsq(A, data[:,2])

    # Copy the original img
    Background = np.ones(img.shape)

    # Fill the bacground with the values came from the fitting
    for X in range(img.shape[0]):
        for Y in range(img.shape[1]):
            Background[X,Y] = C[0] + C[1]*X + C[2]*Y + C[3]*X*Y + C[4]*X*X + C[5]*Y*Y

    return img - Background + Background.mean()


def iou(img1, img2, background_correcting=True, filtering=True):
    """Intersection over union"""
    # correct the backgorund
    if background_correcting:
        img1 = background_correction(img1)
        img2 = background_correction(img2)
        
    # Filter out the background
    if filtering:
        img1 = img1 * (img1 > threshold_otsu(img1))
        img2 = img2 * (img2 > threshold_otsu(img2))
    
    # Comput the intersection and union IOU 
    return np.count_nonzero(img1 * img2) / np.count_nonzero(img1 + img2)