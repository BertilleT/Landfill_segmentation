"""
* GeoTIFF read and write
* extract metadata: datetime, RPC,
* miscelaneous functions for crop
* wrappers for gdaltransform and gdalwarp

Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-paris-saclay.fr>
"""
import os
import datetime
import requests
import subprocess
import numpy as np
import warnings
import pyproj
import rasterio
import geojson
import bs4
import rpcm

warnings.filterwarnings("ignore",
                        category=rasterio.errors.NotGeoreferencedWarning)


def readGTIFF(fname):
    """
    Reads an image file into a numpy array,
    returns the numpy array with dimensios (height, width, channels)
    The returned numpy array is always of type numpy.float
    """
    # read the image into a np.array
    with  rasterio.open(fname, 'r') as s:
        # print('reading image of size: %s'%str(im.shape))
        im = s.read()
    return im.transpose([1,2,0]).astype(np.float)


def readGTIFFmeta(fname):
    """
    Reads the image GeoTIFF metadata using rasterio and returns it,
    along with the bounding box, in a tuple: (meta, bounds)
    if the file format doesn't support metadata the returned metadata is invalid
    This is the metadata rasterio was capable to interpret,
    but the ultimate command for reading metadata is *gdalinfo*
    """
    with  rasterio.open(fname, 'r') as s:
        ## interesting information
        # print(s.crs,s.meta,s.bounds)
        return (s.meta,s.bounds)


def get_driver_from_extension(filename):
    import os.path
    ext = os.path.splitext(filename)[1].upper()
    if ext in ('.TIF', '.TIFF'):
        return 'GTiff'
    elif ext in ('.JPG', '.JPEG'):
        return 'JPEG'
    elif ext == '.PNG':
        return 'PNG'
    return None


def writeGTIFF(im, fname, copy_metadata_from=None):
    """
    Writes a numpy array to a GeoTIFF, PNG, or JPEG image depending on fname extension.
    For GeoTIFF files the metadata can be copied from another file.
    Note that if  im  and  copy_metadata_from have different size,
    the copied geolocation properties are not adapted.
    """
    import rasterio
    import numpy as np

    # set default metadata profile
    p = {'width': 0, 'height': 0, 'count': 1, 'dtype': 'uint8', 'driver': 'PNG',
             'affine': rasterio.Affine (0,1,0,0,1,0),
             'crs': rasterio.crs.CRS({'init': 'epsg:32610'}),
             'nodata': None}

    # read and update input metadata if available
    if copy_metadata_from:
        x = rasterio.open(copy_metadata_from, 'r')
        p.update( x.profile )

    # format input
    if  len(im.shape) == 2:
        im = im[:,:,np.newaxis]

    # override driver and shape
    indriver = get_driver_from_extension(fname)
    if indriver and (indriver != p['driver']):
        #print('writeGTIFF: driver override from %s to %s'%( p['driver'], indriver))
        p['driver'] = indriver or p['driver']
        p['dtype'] = 'float32'

    #if indriver == 'GTiff' and (p['height'] != im.shape[0]  or  p['width'] != im.shape[1]):
    #    # this is a problem only for GTiff
    #    print('writeGTIFF: changing the size of the GeoTIFF')
    #else:
    #    # remove useless properties
    #    p.pop('tiled')

    p['height'] = im.shape[0]
    p['width']  = im.shape[1]
    p['count']  = im.shape[2]

    with rasterio.open(fname, 'w', **p) as d:
            d.write((im.transpose([2,0,1]).astype(d.profile['dtype'])))


def is_absolute(url):
    return bool(requests.utils.urlparse(url).netloc)


def find(url, extension):
    """
    Recursive directory listing, like "find . -name "*extension".

    Args:
        url (str): directory url
        extension (str): file extension to match

    Returns:
        list of urls to files
    """
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    files = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    folders = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('/')]

    files_urls = [f if is_absolute(f) else os.path.join(url, os.path.basename(f)) for f in files]
    folders_urls = [f if is_absolute(f) else os.path.join(url, os.path.basename(f.rstrip('/'))) for f in folders]

    for u in folders_urls:
        if not u.endswith(('../', '..')):
            files_urls += find(u, extension)
    return files_urls


def acquisition_date(geotiff_path):
    """
    Read the image acquisition date in GeoTIFF metadata.

    Args:
        geotiff_path (str): path or url to a GeoTIFF file

    Returns:
        datetime.datetime object with the image acquisition date
    """
    with rasterio.open(geotiff_path, 'r') as src:
        if 'NITF_IDATIM' in src.tags():
            date_string = src.tags()['NITF_IDATIM']
        elif 'NITF_STDIDC_ACQUISITION_DATE' in src.tags():
            date_string = src.tags()['NITF_STDIDC_ACQUISITION_DATE']
        return datetime.datetime.strptime(date_string, "%Y%m%d%H%M%S")


def gdal_get_longlat_of_pixel(fname, x, y, verbose=True):
    """
    returns the longitude latitude and altitude (wrt the WGS84 reference
    ellipsoid) for the points at pixel coordinates (x, y) of the image fname.
    The CRS of the input GeoTIFF is determined from the metadata in the file.

    """
    import subprocess

    # add vsicurl prefix if needed
    env = os.environ.copy()
    if fname.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = fname[-3:]
        fname = '/vsicurl/{}'.format(fname)

    # form the query string for gdaltransform
    q = b''
    for (xi,yi) in zip(x,y):
        q = q + b'%d %d\n'%(xi,yi)
    # call gdaltransform, "+proj=longlat" uses the WGS84 ellipsoid
    #    echo '0 0' | gdaltransform -t_srs "+proj=longlat" inputimage.tif
    cmdlist = ['gdaltransform', '-t_srs', "+proj=longlat", fname]
    if verbose:
        print ('RUN: ' +  ' '.join(cmdlist) + ' [x y from stdin]')
    p = subprocess.Popen(cmdlist,
                         stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    out = (p.communicate(q)[0]).decode()
    listeout =  [ list(map(float, x.split())) for x in out.splitlines()]
    return listeout


def lon_lat_image_footprint(image, z=0):
    """
    Compute the longitude, latitude footprint of an image using its RPC model.

    Args:
        image (str): path or url to a GeoTIFF file
        z (float): altitude (in meters above the WGS84 ellipsoid) used to
            convert the image corners pixel coordinates into longitude, latitude

    Returns:
        geojson.Polygon object containing the image footprint polygon
    """
    rpc = rpcm.rpc_from_geotiff(image)
    with rasterio.open(image, 'r') as src:
        h, w = src.shape
    coords = []
    for x, y, z in zip([0, w, w, 0], [0, 0, h, h], [z, z, z, z]):
        lon, lat = rpc.localization(x, y, z)
        coords.append([lon, lat])
    return geojson.Polygon([coords])


def gdal_resample_image_to_longlat(fname, outfname, verbose=True):
    """
    resample a geotiff image file in longlat coordinates (EPSG: 4326 with WGS84 datum)
    and saves the result in outfname
    """
    import os

    driver = get_driver_from_extension(outfname)
    cmd = 'gdalwarp -overwrite  -of %s -t_srs "+proj=longlat +datum=WGS84" %s %s'%(driver, fname, outfname)
    if verbose:
        print('RUN: ' + cmd)
    return os.system(cmd)


def bounding_box2D(pts):
    """
    Rectangular bounding box for a list of 2D points.

    Args:
        pts (list): list of 2D points represented as 2-tuples or lists of length 2

    Returns:
        x, y, w, h (floats): coordinates of the top-left corner, width and
            height of the bounding box
    """
    if type(pts) == list  or  type(pts) == tuple:
        pts = np.array(pts).squeeze()
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]


def image_crop_gdal(inpath, x, y, w, h, outpath):
    """
    Image crop defined in pixel coordinates using gdal_translate.

    Args:
        inpath: path to an image file
        x, y, w, h: four integers defining the rectangular crop pixel coordinates.
            (x, y) is the top-left corner, and (w, h) are the dimensions of the
            rectangle.
        outpath: path to the output crop
    """
    if int(x) != x or int(y) != y:
        print('WARNING: image_crop_gdal will round the coordinates of your crop')

    env = os.environ.copy()
    if inpath.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = inpath[-3:]
        path = '/vsicurl/{}'.format(inpath)
    else:
        path = inpath

    cmd = ['gdal_translate', path, outpath,
           '-srcwin', str(x), str(y), str(w), str(h),
           '-ot', 'Float32',
           '-co', 'TILED=YES',
           '-co', 'BIGTIFF=IF_NEEDED']

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        if inpath.startswith(('http://', 'https://')):
            if not requests.head(inpath).ok:
                print('{} is not available'.format(inpath))
                return
        print('ERROR: this command failed')
        print(' '.join(cmd))
        print(e.output)


def points_apply_homography(H, pts):
    """
    Applies an homography to a list of 2D points.

    Args:
        H (np.array): 3x3 homography matrix
        pts (list): list of 2D points, each point being a 2-tuple or a list
            with its x, y coordinates

    Returns:
        a numpy array containing the list of transformed points, one per line
    """
    pts = np.asarray(pts)

    # convert the input points to homogeneous coordinates
    if len(pts[0]) < 2:
        print("""points_apply_homography: ERROR the input must be a numpy array
          of 2D points, one point per line""")
        return
    pts = np.hstack((pts[:, 0:2], pts[:, 0:1]*0+1))

    # apply the transformation
    Hpts = (np.dot(H, pts.T)).T

    # normalize the homogeneous result and trim the extra dimension
    Hpts = Hpts * (1.0 / np.tile( Hpts[:, 2], (3, 1)) ).T
    return Hpts[:, 0:2]


def bounding_box_of_projected_aoi(rpc, aoi, z=0, homography=None):
    """
    Return the x, y, w, h pixel bounding box of a projected AOI.

    Args:
        rpc (rpcm.RPCModel): RPC camera model
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI
        z (float): altitude of the AOI with respect to the WGS84 ellipsoid
        homography (2D array, optional): matrix of shape (3, 3) representing an
            homography to be applied to the projected points before computing
            their bounding box.

    Return:
        x, y (ints): pixel coordinates of the top-left corner of the bounding box
        w, h (ints): pixel dimensions of the bounding box
    """
    lons, lats = np.array(aoi['coordinates'][0]).T
    x, y = rpc.projection(lons, lats, z)
    pts = list(zip(x, y))
    if homography is not None:
        pts = points_apply_homography(homography, pts)
    return np.round(bounding_box2D(pts)).astype(int)


def crop_aoi(geotiff, aoi, z=0):
    """
    Crop a geographic AOI in a georeferenced image using its RPC functions.

    Args:
        geotiff (string): path or url to the input GeoTIFF image file
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI
        z (float, optional): base altitude with respect to WGS84 ellipsoid (0
            by default)

    Return:
        crop (array): numpy array containing the cropped image
        x, y, w, h (ints): image coordinates of the crop. x, y are the
            coordinates of the top-left corner, while w, h are the dimensions
            of the crop.
    """
    x, y, w, h = bounding_box_of_projected_aoi(rpcm.rpc_from_geotiff(geotiff), aoi, z)
    with rasterio.open(geotiff, 'r') as src:
        crop = src.read(window=((y, y + h), (x, x + w)), boundless=True).squeeze()
    return crop, x, y


def lonlat_to_utm(lons, lats, force_epsg=None):
    """
    Convert longitude, latitude to UTM coordinates.

    Args:
        lons (float or list): longitude, or list of longitudes
        lats (float or list): latitude, or list of latitudes
        force_epsg (int): optional EPSG code of the desired UTM zone

    Returns:
        eastings (float or list): UTM easting coordinate(s)
        northings (float or list): UTM northing coordinate(s)
        epsg (int): EPSG code of the UTM zone
    """
    lons = np.atleast_1d(lons)
    lats = np.atleast_1d(lats)
    if force_epsg:
        epsg = force_epsg
    else:
        epsg = compute_epsg(lons[0], lats[0])
    e, n = pyproj_lonlat_to_epsg(lons, lats, epsg)
    return e.squeeze(), n.squeeze(), epsg


def utm_to_lonlat(eastings, northings, epsg):
    """
    Convert UTM coordinates to longitude, latitude.

    Args:
        eastings (float or list): UTM easting coordinate(s)
        northings (float or list): UTM northing coordinate(s)
        epsg (int): EPSG code of the UTM zone

    Returns:
        lons (float or list): longitude, or list of longitudes
        lats (float or list): latitude, or list of latitudes
    """
    eastings = np.atleast_1d(eastings)
    northings = np.atleast_1d(northings)
    lons, lats = pyproj_epsg_to_lonlat(eastings, northings, epsg)
    return lons.squeeze(), lats.squeeze()


def utm_zone_to_epsg(utm_zone, northern_hemisphere=True):
    """
    Args:
        utm_zone (int):
        northern_hemisphere (bool): True if northern, False if southern

    Returns:
        epsg (int): epsg code
    """
    # EPSG = CONST + ZONE where CONST is
    # - 32600 for positive latitudes
    # - 32700 for negative latitudes
    const = 32600 if northern_hemisphere else 32700
    return const + utm_zone


def epsg_to_utm_zone(epsg):
    """
    Args:
        epsg (int):

    Returns:
        utm_zone (int): zone number
        northern_hemisphere (bool): True if northern, False if southern
    """
    if (32600 < epsg <= 32660):
        return epsg % 100, True
    elif (32700 < epsg <= 32760):
        return epsg % 100, False
    else:
        raise Exception("Invalid UTM epsg code: {}".format(epsg))


def compute_epsg(lon, lat):
    """
    Compute the EPSG code of the UTM zone which contains
    the point with given longitude and latitude

    Args:
        lon (float): longitude of the point
        lat (float): latitude of the point

    Returns:
        int: EPSG code
    """
    # UTM zone number starts from 1 at longitude -180,
    # and increments by 1 every 6 degrees of longitude
    zone = int((lon + 180) // 6 + 1)

    return utm_zone_to_epsg(zone, lat > 0)


def pyproj_transform(x, y, in_crs, out_crs, z=None):
    """
    Wrapper around pyproj to convert coordinates from an EPSG system to another.

    Args:
        x (scalar or array): x coordinate(s), expressed in in_crs
        y (scalar or array): y coordinate(s), expressed in in_crs
        in_crs (pyproj.crs.CRS or int): input coordinate reference system or EPSG code
        out_crs (pyproj.crs.CRS or int): output coordinate reference system or EPSG code
        z (scalar or array): z coordinate(s), expressed in in_crs

    Returns:
        scalar or array: x coordinate(s), expressed in out_crs
        scalar or array: y coordinate(s), expressed in out_crs
        scalar or array (optional if z): z coordinate(s), expressed in out_crs
    """
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    if z is None:
        return transformer.transform(x, y)
    else:
        return transformer.transform(x, y, z)


def pyproj_lonlat_to_epsg(lon, lat, epsg):
    return pyproj_transform(lon, lat, 4326, epsg)


def pyproj_epsg_to_lonlat(x, y, epsg):
    return pyproj_transform(x, y, epsg, 4326)


def pyproj_lonlat_to_utm(lon, lat, epsg=None):
    if epsg is None:
        epsg = compute_epsg(np.mean(lon), np.mean(lat))
    x, y = pyproj_lonlat_to_epsg(lon, lat, epsg)
    return x, y, epsg



def utm_bounding_box_from_lonlat_aoi(aoi):
    """
    Computes the UTM bounding box (min_easting, min_northing, max_easting,
    max_northing)  of a projected AOI.

    Args:
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI expressed in (long, lat)

    Return:
        min_easting, min_northing, max_easting, max_northing: the coordinates
        of the top-left corner and lower-right corners of the aoi in UTM coords
    """
    lons, lats  = np.array(aoi['coordinates'][0]).T
    east, north, zone = lonlat_to_utm(lons, lats)
    pts = list(zip(east, north))
    emin, nmin, deltae, deltan = bounding_box2D(pts)
    return emin, emin+deltae, nmin, nmin+deltan


def simple_equalization_8bit(im, percentiles=5):
    """
    Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (int): percentage of the darkest and brightest pixels to saturate

    Returns:
        numpy array with the quantized uint8 image
    """
    import numpy as np
    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles, 100 - percentiles))
    im = np.clip(im, mi, ma)
    im = (im - mi) / (ma - mi) * 255   # scale
    return im.astype(np.uint8)


def simplest_color_balance_8bit(im, percentiles=5):
    """
    Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (int): percentage of the darkest and brightest pixels to saturate

    Returns:
        numpy array with the quantized uint8 image
    """
    import numpy as np
    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles, 100 - percentiles))
    im = np.clip(im, mi, ma)
    im = (im - mi) / (ma - mi) * 255   # scale
    return im.astype(np.uint8)


def matrix_translation(x, y):
    """
    Return the (3, 3) matrix representing a 2D shift in homogeneous coordinates.
    """
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def get_angle_from_cos_and_sin(c, s):
    """
    Computes x in ]-pi, pi] such that cos(x) = c and sin(x) = s.
    """
    if s >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)


def find_key_in_geojson(it,q):
    """
    Traverses the geojson object it (dictionary mixed with lists) 
    in a depth first order and returns the first ocurrence of the 
    keywork q, otherwise returns None
    example: 
        aoi = find_key_in_geojson(geojson.loads(geojsonstring),'geometry')
    """
    if (isinstance(it, list)):
        for item in it:
            t = find_key_in_geojson(item,q)
            if  t is not None:
                return t
        return None
    elif (isinstance(it, dict)):
        if q in it.keys():
            return it[q]
        for key in it.keys():
            t = find_key_in_geojson(it[key],q)
            if  t is not None:
                return  t
        return None 

    else:
        return None



