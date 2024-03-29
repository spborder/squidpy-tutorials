"""

Reading non-kidney data and finding all necessary stuff for creating annotations in FUSION

"""

import os
import sys
import numpy as np

import spatialdata as sd
import spatialdata_plot
import anndata as ad
import large_image

from shapely.geometry import Point
import geopandas as gpd

import pandas as pd


def calculate_mpp(coordinates_array):

    # Finding minimum distance spots from first spot and using that to determine MPP
    # spot centroids are 100um apart and spot diameters are 55um
    spot_x_coords = coordinates_array[:,0]
    spot_y_coords = coordinates_array[:,1]

    # Test spot is the first one
    test_spot = [spot_x_coords[0],spot_y_coords[0]]

    # Spot distances
    spot_dists = np.array([distance(test_spot, [x, y]) for x, y in zip(spot_x_coords, spot_y_coords)])
    spot_dists = np.unique(spot_dists[spot_dists > 0])
    min_spot_dist = np.min(spot_dists)

    # Minimum distance between the test spot and another spot = 100um (same as doing 100/min_spot_dist)
    mpp = 1/(min_spot_dist/100)

    return mpp

def distance(point1, point2):
    # Distance between 2 points
    return (((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2))**0.5




def main():
    path_to_data = 'C:\\Users\\samuelborder\\Desktop\\HIVE_Stuff\\FUSION\\Test Upload\\non_kidney\\'
    print(os.listdir(path_to_data))

    ann_data_object = ad.read_h5ad(path_to_data+'secondary_analysis.h5ad')
    print(ann_data_object)
    print(ann_data_object.n_obs)
    print(ann_data_object.n_vars)
    print(ann_data_object.shape)
    print(ann_data_object.uns["spatial"])
    print(ann_data_object.obsm['spatial'])
    print('--------------------------')
    print(ann_data_object.uns['umap'])
    print(ann_data_object.obsm['X_umap'])
    print(ann_data_object.var['highly_variable'].sum())
    print(list(ann_data_object.var['mean'].sort_values(ascending=False).iloc[0:10].index))
    highest_means = list(ann_data_object.var['mean'].sort_values(ascending=False).iloc[0:10].index)
    subset_ann_data = ann_data_object[:,highest_means]
    print(subset_ann_data)
    print(subset_ann_data.var['dispersions'])
    print('------------------------------------')

    image_source = large_image.open(path_to_data+'visium_histology_hires_pyramid.ome.tif')
    image_meta = image_source.getMetadata()
    print(image_meta)

    image_combined_array = np.zeros((3,image_meta['sizeY'],image_meta['sizeX']),dtype=np.uint8)
    for i in range(len(image_meta['frames'])):
        image_array,_ = image_source.getRegion(
            format = large_image.constants.TILE_FORMAT_NUMPY,
            region = {
                'left': 0,
                'top': 0,
                'right': image_meta['sizeX'],
                'bottom': image_meta['sizeY'],
                'frame': i
            }
        )
        image_combined_array[i,:,:] += np.uint8(np.squeeze(image_array))

    # Generating shapes:
    centers_array = ann_data_object.obsm['spatial']
    shape_list = []
    # Not sure what they mean by spot_diameter_fullres, this number is way too large
    #spot_radius = ann_data_object.uns['spatial']['visium']['scalefactors']['spot_diameter_fullres']/2
    mpp = calculate_mpp(centers_array)
    spot_pixel_diameter = int((1/mpp)*55)
    spot_pixel_radius = int(spot_pixel_diameter/2)
    print(f'Calculated spot radius: {spot_pixel_radius}')
    for spot in centers_array.tolist():

        spot_circle = Point(spot[0],spot[1]).buffer(spot_pixel_radius)
        shape_list.append(spot_circle)
    
    shape_gdf = gpd.GeoDataFrame(geometry = shape_list)

    non_kidney_sd = sd.SpatialData(
        images = {
            'non_kidney_image':sd.models.Image2DModel.parse(
                image_combined_array
            )
        },
        tables = {
            'non_kidney_table': ann_data_object
        },
        shapes = {
            'visium_spots': sd.models.ShapesModel.parse(shape_gdf)
        }
    )
    print(non_kidney_sd)

    spot_df = pd.DataFrame(
        data = ann_data_object.obsm['spatial'],
        index = ann_data_object.obs_names,
        columns = ['imagecol','imagerow']
    )
    print(spot_df.head(10))

    non_kidney_sd.pl.render_images().pl.render_shapes().pl.show('global')



if __name__=='__main__':
    main()





















