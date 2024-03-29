import rasterio
import os
import numpy as np
import json
import subprocess
import cv2
import matplotlib.pyplot as plt


"""
In this script, we suppose that we already ran the script query_images.py to get the images in tif format.
Now the goal is to create a dataset from the images we have (we only take the images that have a resolution
or more than 20000 pixels). We also create ground truth masks from the polygons in the geojson.
"""

def gdal_get_pixel_of_longlat(fname, longlat, verbose=True):
    """
    Thid function is the inverse of gdal_get_longlat_of_pixel that can be found in utils.py (from TP1)
    returns the pixel coordinates for the points at longitude and latitude in the GeoTIFF image fname. 
    The CRS of the input GeoTIFF is determined from the metadata in the file.
    """
    # add vsicurl prefix if needed
    env = os.environ.copy()
    if fname.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = fname[-3:]
        fname = '/vsicurl/{}'.format(fname)

    # form the query string for gdaltransform
    q = b''
    for coords in longlat:
        if len(coords) == 2:  # Only longitude and latitude provided
            coords = (*coords, 0)  # Append a default altitude of 0
        q = q + b'%f %f %f\n' % coords  # Unpack the tuple directly

    # call gdaltransform to convert longlat to pixel coordinates, -i does the inverse transformation
    cmdlist = ['gdaltransform', '-t_srs', "+proj=longlat", '-i', fname]
    if verbose:
        print('RUN: ' + ' '.join(cmdlist) + ' [long lat [alt] from stdin]')
    p = subprocess.Popen(cmdlist, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(q)
    if p.returncode != 0:
        raise RuntimeError("Command failed: %s\nError message: %s" % (' '.join(cmdlist), err.decode()))

    # parse the output from gdaltransform to get pixel coordinates
    listeout = [list(map(float, x.split())) for x in out.decode().splitlines()]
    # remove last coordinate of each point (which is the altitude)
    listeout = [x[:2] for x in listeout]
    return listeout


def create_mask_from_polygon(image_shape, polygon):
    """
    Create a binary mask from a polygon.

    Parameters:
        image_shape (tuple): The shape of the image (height, width).
        polygon (list): A list of lists, where each inner list contains the [x, y] coordinates of a vertex.

    Returns:
        np.array: A binary mask with the same dimensions as the image.
    """
    poly_array = np.array(polygon, dtype=np.int32)

    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Fill the polygon with ones
    cv2.fillPoly(mask, [poly_array], color=(1))

    return mask


# pipeline to create the dataset with images and ground truth masks
def create_dataset(imgs_path, dataset_path, keep_images_ids, keep_np_images, data):
    """_summary_

    Args:
        path (_type_): the path where to save the dataset (images and masks)
        keep_images_ids (_type_): the ids of the images in the geojson to retrieve the corresponding polygons
        keep_np_images (_type_): the numpy images
        data (_type_): the geojson data
    """
    # create folders for the images and masks
    os.makedirs(dataset_path + 'images', exist_ok=True)
    os.makedirs(dataset_path + 'masks', exist_ok=True)

    for i in range(len(keep_images_ids)):
        try:
            idx = keep_images_ids[i]
            fname = imgs_path + f'{idx}.tif'
            polygon = data['features'][idx]['geometry']['coordinates'][0]
            # transform the polygon to pixel coordinates
            polygon = gdal_get_pixel_of_longlat(fname, polygon)
            mask = create_mask_from_polygon(keep_np_images[i].shape[:2], polygon)
            # crop mask and image to a square centering as much as possible the polygon
            # check if x or y axis is the longest
            if mask.shape[0] > mask.shape[1]:
                # crop the image
                image = keep_np_images[i][int(mask.shape[0]/2 - mask.shape[1]/2):int(mask.shape[0]/2 + mask.shape[1]/2)]
                # crop the mask
                mask = mask[int(mask.shape[0]/2 - mask.shape[1]/2):int(mask.shape[0]/2 + mask.shape[1]/2)]
            else:
                # crop the image
                image = keep_np_images[i][:, int(mask.shape[1]/2 - mask.shape[0]/2):int(mask.shape[1]/2 + mask.shape[0]/2)]
                # crop the mask
                mask = mask[:, int(mask.shape[1]/2 - mask.shape[0]/2):int(mask.shape[1]/2 + mask.shape[0]/2)]
            # transform image from uint16 to uint8
            image = image.astype(np.uint8)
            # save the mask
            plt.imsave(dataset_path + f'images/image_{idx}.png', image)
            plt.imsave(dataset_path + f'masks/mask_{idx}.png', mask, cmap='gray')
        except:
            print(f'Error with image {keep_images_ids[i]}')
        if i % 100 == 0:
            print(f'{i} images processed')

if __name__ == "__main__":
    # path of the tif images
    path = 'imgs/'
    
    images_names = [f for f in os.listdir(path) if f.endswith('.tif')]
    # the ids of the images in the geojson to retrieve the corresponding polygons
    images_ids = [int(image.split('large')[1].split('.')[0]) for image in images_names]
    images_tif = [rasterio.open(path + image) for image in images_names]
    np_images = [np.moveaxis(image.read(), 0, -1) for image in images_tif]
    nb_pixels = [image.shape[0] * image.shape[1] for image in np_images]
    
    # keep ids of the images with more than 20000 pixels
    keep_images_ids = [images_ids[i] for i in range(len(images_ids)) if nb_pixels[i] > 20000]
    keep_np_images = [np_images[i] for i in range(len(np_images)) if nb_pixels[i] > 20000]
    
    # change the path if needed
    with open('landfills_openstreetmap.geojson') as f:
        data = json.load(f)
        
    # create the dataset
    create_dataset('imgs/', 'dataset/', keep_images_ids, keep_np_images, data)
        
    
    
    