"""

Using SpatialData to align CODEX dataset from DSA

"""

import os
import sys

import numpy as np

from shapely.geometry import Polygon, Point, box
import tiffslide as ts
import large_image

import spatialdata as sd
import spatialdata_plot

import geopandas as gpd

from io import BytesIO
from PIL import Image
from math import floor
import json

def shapely_from_json(json_annotations, scale_list):

    shape_list = []
    for el in json_annotations['annotation']['elements']:

        if el['type']=='point':
            shape_list.append(
                Point(el['center'][0]/scale_list[0],el['center'][1]/scale_list[0])
            )
    
    return shape_list

def postpone_transformation(
    sdata: sd.SpatialData,
    transformation: sd.transformations.BaseTransformation,
    source_coordinate_system: str,
    target_coordinate_system: str,
):
    for element_type, element_name, element in sdata._gen_elements():
        old_transformations = sd.transformations.get_transformation(element, get_all=True)
        if source_coordinate_system in old_transformations:
            old_transformation = old_transformations[source_coordinate_system]
            sequence = sd.transformations.Sequence([old_transformation, transformation])
            sd.transformations.set_transformation(element, sequence, target_coordinate_system)

def labels_from_shapely(shape_list, target_size_Y, target_size_X):
    """
    Create a label image from a list of shapes
    """
    from skimage.draw import polygon

    label_mask = np.zeros((target_size_Y,target_size_X))
    for shape_idx,shape in enumerate(shape_list):
        if shape.geom_type == 'Polygon':
            coords = list(shape.exterior.coords)
            x_coords = [i[0] for i in coords]
            y_coords = [i[1] for i in coords]

            # Verify type requirements for this (what if one crs features floats?)
            bounds = [int(i) for i in shape.bounds]

            height = bounds[3]-bounds[1]
            width = bounds[2]-bounds[0]
            scaled_x_coords = [i-bounds[0] for i in x_coords]
            scaled_y_coords = [i-bounds[1] for i in y_coords]

            shape_mask = np.zeros((height,width))
            row, col = polygon(r = scaled_y_coords, c = scaled_x_coords)
            shape_mask[row,col] = shape_idx+1

            label_mask[bounds[1]:bounds[3], bounds[0]:bounds[2]] += shape_mask

        else:
            print(f'Invalid polygon type found: {shape.geom_type}')
            continue

    return label_mask








def main():

    # Reading images (Have to load the whole image as a numpy array in order to initialize )
    histology_img = large_image.open('./dsa/15-1 Stitch.svs')
    histo_name = '15-1 Stitch.svs'
    #histoId = '6461bbe818a9d4c4175edbcd'
    histo_meta = histology_img.getMetadata()
    histo_sizeX, histo_sizeY = histo_meta['sizeX'], histo_meta['sizeY']

    histo_thumbnail, _ = histology_img.getThumbnail()
    histo_thumbnail = np.array(Image.open(BytesIO(histo_thumbnail)))
    histo_thumbnail_shape = np.shape(histo_thumbnail)
    histo_thumbnail = np.moveaxis(histo_thumbnail,source=-1, destination = 0)

    # 
    codex_img = large_image.open('./dsa/15-1.tif')
    codex_name = '15-1.tif'
    #codexId = '648b46573e6ae3107da0d990'
    codex_meta = codex_img.getMetadata()
    codex_sizeX, codex_sizeY = codex_meta['sizeX'], codex_meta['sizeY']

    codex_thumbnail, _ = codex_img.getThumbnail()
    codex_thumbnail = np.array(Image.open(BytesIO(codex_thumbnail)))    
    codex_thumbnail_shape = np.shape(codex_thumbnail)
    codex_thumbnail = np.moveaxis(codex_thumbnail, source = -1, destination = 0)
    # Reading smaller size level into memory

    histo_scale_X = histo_sizeX/histo_thumbnail_shape[1]
    histo_scale_Y = histo_sizeY/histo_thumbnail_shape[0]
    codex_scale_X = codex_sizeX/codex_thumbnail_shape[1]
    codex_scale_Y = codex_sizeY/codex_thumbnail_shape[0]

    
    # Reading landmark locations for each image
    with open('./dsa/histo_landmarks.json','r') as f:
        histo_json = json.load(f)
        f.close()

    histo_landmarks = shapely_from_json(histo_json, [histo_scale_X, histo_scale_Y])

    with open('./dsa/codex_landmarks.json','r') as f:
        codex_json = json.load(f)
        f.close()

    codex_landmarks = shapely_from_json(codex_json, [codex_scale_X, codex_scale_Y])

    histo_sdata = sd.SpatialData(
        images = {
            'histo_img': sd.models.Image2DModel.parse(histo_thumbnail)
        },
        shapes = {
            'histo_landmarks': sd.models.ShapesModel.parse(
                np.squeeze(np.array([list(i.coords) for i in histo_landmarks])), 
                geometry = 0, 
                radius = 5
            )
        }
    )
    histo_sdata.rename_coordinate_systems({'global':'histo_crs'})
    print(histo_sdata)

    codex_sdata = sd.SpatialData(
        images = {
            'codex_img': sd.models.Image2DModel.parse(codex_thumbnail)
        },
        shapes = {
            'codex_landmarks': sd.models.ShapesModel.parse(
                np.squeeze(np.array([list(i.coords) for i in codex_landmarks])), 
                geometry = 0,
                radius = 5,
            )
        }
    )
    codex_sdata.rename_coordinate_systems({'global':'codex_crs'})
    print(codex_sdata)

    multi_sdata = sd.concatenate([histo_sdata,codex_sdata])
    print(multi_sdata)

    multi_sdata.pl.render_images().pl.render_shapes().pl.show('histo_crs')
    multi_sdata.pl.render_images().pl.render_shapes().pl.show('codex_crs')


    # Aligning based on landmarks
    transform = sd.transformations.align_elements_using_landmarks(
        references_coords = multi_sdata.shapes['histo_landmarks'],
        moving_coords = multi_sdata.shapes['codex_landmarks'],
        reference_element = multi_sdata.images['histo_img'],
        moving_element = multi_sdata.images['codex_img'],
        reference_coordinate_system = 'histo_crs',
        moving_coordinate_system = 'codex_crs',
        new_coordinate_system = 'aligned'
    )
    print(transform)


    postpone_transformation(
        sdata = multi_sdata,
        transformation = transform,
        source_coordinate_system = 'codex_crs',
        target_coordinate_system = 'aligned'
    )

    print(multi_sdata)


    multi_sdata.pl.render_images().pl.render_shapes().pl.show('aligned')
    multi_sdata.pl.render_images('histo_img').pl.render_shapes().pl.show('aligned')
    multi_sdata.pl.render_images('codex_img').pl.render_shapes().pl.show('aligned')



if __name__=='__main__':
    main()












