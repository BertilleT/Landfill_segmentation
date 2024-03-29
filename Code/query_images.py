# These are the main includes used through the notebook
import datetime
import geojson
import numpy as np                   # numeric linear algebra
import matplotlib.pyplot as plt      # plotting
import rasterio       # read/write geotiffs
import tsd            # time series downloader for sentinel and landsat
import utils          # IO and coordinate system conversion tools
import vistools       # display tools
import folium
import folium.plugins
import os
import json

###########################################
### The code is mostly taken from TP1 #####
###########################################

def get_crop_from_aoi(output_path, aoi, catalog_entry, band):
    """
    Crop and download an expanded AOI from a georeferenced image. The expanded AOI will be twice
    as large as the original area defined by the geojson polygon.
    Args:
        output_path (string): path to the output GeoTIFF file
        aoi (geojson.Polygon): area of interest defined by a polygon in longitude, latitude coordinates
        catalog_entry (tsd.s2_metadata_parser.Sentinel2Image): metadata object
        band (str): desired band, e.g. 'B04' for Sentinel-2 or 'B8' for Landsat-8
    """
    metadata = catalog_entry
    if not metadata.urls['aws']:
        metadata.build_s3_links()
    inpath = metadata.urls['aws'][band]

    utm_zone = metadata.get("utm_zone")
    lat_band = metadata.get("lat_band")
    epsg = tsd.utils.utm_to_epsg_code(utm_zone, lat_band) if utm_zone else None

    # Original bounding box
    ulx, uly, lrx, lry, epsg = tsd.utils.utm_bbx(aoi, epsg=epsg, r=60)

    # Modify bounding box to be twice as large
    # Calculate the center of the original bounding box
    center_x = (ulx + lrx) / 2
    center_y = (uly + lry) / 2
    # Calculate the new bounding box coordinates to make the box twice as large
    new_ulx = ulx - (center_x - ulx)
    new_uly = uly - (center_y - uly)
    new_lrx = lrx + (lrx - center_x)
    new_lry = lry + (lry - center_y)

    # Use the expanded AOI for cropping
    tsd.utils.rasterio_geo_crop(output_path, inpath, new_ulx, new_uly, new_lrx, new_lry, epsg=epsg)


def simple_equalization_8bit(im, percentiles=5):
    ''' im is a numpy array
        returns a numpy array
    '''

    ###     <<< CODE HERE >>>
    low = np.percentile(im, percentiles)
    high = np.percentile(im, 100-percentiles)

    im = np.clip(im, low, high)
    im = (im - low) / (high - low)
    im = np.uint8(im * 255)

    return im


def query_clear_sky(aoi, satellite='Sentinel-2', max_cloud_cover=30,
                    start_date=None, end_date=None):
    '''
    queries the devseed database for the aoi
    returns a filtered catalog with images with cloud
    cover of at most max_cloud_cover
    '''
    # run a search
    if satellite == 'Sentinel-2':
        res = tsd.get_sentinel2.search(aoi, start_date, end_date)
    elif satellite == 'Landsat-8':
        res = tsd.get_landsat.search(aoi, start_date, end_date)


    ###### Insert your solution code here ######
    res2 = []
    for image in res:
      if image.cloud_cover <= max_cloud_cover:
          res2.append(image)

    return res2


def get_sentinel2_color_8bit(basefilename, aoi, catalog_entry, percentiles=5):
    ''' basefilename to store the bands:  basename+BAND+'.tif'
        returns a numpy array of the RGB image (height, width, channels)
    '''
    bands = ['B04', 'B03', 'B02']    # SENTINEL2 R,G,B BANDS

    # this command downloads all the bands
    for b in bands:
        get_crop_from_aoi('{}_{}.tif'.format(basefilename, b), aoi, catalog_entry, b)

    # read, equalize, and stack all the channels
    out = []
    for b in bands:
        im = utils.readGTIFF('{}_{}.tif'.format(basefilename, b))
        im = simple_equalization_8bit(im, percentiles)
        out.append(im)

    # The transposition is necessary because the indexing
    # convention for color images is (height, width, channels)
    im = np.squeeze(out,axis=(3)).transpose(1,2,0)
    return im


def get_sentinel2_image(geojsonstring, start_date, end_date, max_cloud_cover=30, percentiles=5, satellite='Sentinel-2', path='.'):
    ''' returns a numpy array of the RGB image (height, width, channels)
    '''
    # pick the satellite then query the database
    aoi = utils.find_key_in_geojson(geojson.loads(geojsonstring),'geometry')
    basename = 'rgb'
    res = query_clear_sky(aoi, satellite, max_cloud_cover, start_date, end_date)

    # generate the RGB image
    RGBout = get_sentinel2_color_8bit(basename, aoi, res[-1], percentiles)

    # Writes RGBout in 'rgb_RGB.tif' copying geolocation metadata from 'rgb_B03.tif',
    # which has been written by    get_sentinel2_color_8bit
    utils.writeGTIFF(RGBout, path+'.tif', basename+'_B03.tif')
    
    # delete the intermediate files
    for b in ['B02', 'B04', 'B03']:
        os.remove('{}_{}.tif'.format(basename, b))

    return RGBout

def main(imgs_path):
    with open('landfills_openstreetmap.geojson') as f:
        data = json.load(f)
    
    start_date = datetime.datetime(2023, 7, 1)
    end_date = datetime.datetime(2023, 12, 31)
    for i in range(len(data['features'][:10000])):
        try :
            geojsonstring = str(data['features'][i])
            geojsonstring = geojsonstring.replace("'", '"')
            # savew the image in the imgs_path folder in the tif format
            RGBout = get_sentinel2_image(geojsonstring, start_date=start_date, end_date=end_date, max_cloud_cover=10, percentiles=5, satellite='Sentinel-2', path=imgs_path+str(i))
            if i%10 == 0:
                print(str(i), 'done')
        except:
            print(str(i), 'failed')
            continue
    return

if __name__ == "__main__":
    imgs_path = 'imgs/'
    main(imgs_path)
    
    