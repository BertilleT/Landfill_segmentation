# Landfill Segmentation
Remote Sensing Data project.  
Students: Rim El Mallah and Bertille Temple.

## Organisation Tree

### `Report.pdf`
This PDF file is the report made for the project of the MVA course "Remote Sensing Data," delivered in April 2024.

### `Data`
This folder is intended for storing the project data.
- `geojson`: Contains the GeoJSON files extracted from OpenStreetMap (provided by the course instructors).
- `img_mask`: This folder is divided into two sections:
    - `images` for the downloaded Sentinel 2 (S2) images.
    - `masks` for the masks generated from the OSM polygons and S2 images.

### `Models`
Here, trained models are stored, such as the model trained with UNet.

### `Notebooks`
This section includes Jupyter notebooks used in the project, listed as follows:
- `explore_data_notebook.ipynb`: script to explore the data in the geojson files provided. 
- `Unet.ipynb`: script to predict masks with Unet model. Can be run in Google Collab. 
- 
