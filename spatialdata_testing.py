"""

Testing out SpatialData


"""

import os
import sys

import numpy as np
import geojson
import geopandas as gpd

from shapely.geometry import Polygon, box, Point, MultiPolygon

import spatialdata as sd
import spatialdata_plot

from math import floor

def generate_random_boxes(image_Y,image_X,n_boxes=200,max_size=100):
    """
    Generate random box polygons for a given image. Outputs labeled mask that is the same spatial dimensions as the original image (image_Y,image_X)
    
    Does not check if boxes overlap
    """
    #box_mask = np.zeros((image_Y,image_X))
    box_list = []

    for b in range(n_boxes):

        # Random box center:
        box_center_y = np.random.randint(low=0,high=image_Y)
        box_center_x = np.random.randint(low=0,high=image_X)

        # Generating height and width:
        box_dim = np.random.randint(low=1,high = floor(max_size/2))
        top = np.maximum(0,box_center_y - box_dim)
        left = np.maximum(0,box_center_x - box_dim)
        bottom = np.minimum(image_Y,box_center_y + box_dim)
        right = np.minimum(image_X, box_center_x + box_dim)

        box_poly = box(left, top, right, bottom)

        # Creating labeled mask
        #box_mask[top:bottom, left:right] = b+1
        box_list.append(box_poly)
    
    return box_list

def get_landmarks(bounds, size = 10):
    """
    Creating landmarks to use to align between global and specific crs    
    """

    top_left_box = Point(bounds[0],bounds[1])

    bottom_right_box = Point(bounds[2],bounds[3])
    return [top_left_box,bottom_right_box]

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




def main():

    visium_sdata = sd.read_zarr('./visium/data.zarr')
    print(visium_sdata)
    image_name = 'ST8059048'

    # visiualization with shapes
    #visium_sdata.pl.render_images().pl.render_shapes().pl.show(image_name)

    # Visualization of gene expression per-spot
    visium_sdata.table.to_df().sum(axis=0).sort_values(ascending=False).head(10)
    # We will select some of the highly expressed genes for this example
    
    #(visium_sdata.pp.get_elements(image_name).pl.render_images().pl.render_shapes(color="mt-Co3").pl.show())

    # Creating the random box annotations
    c, shape_Y, shape_X = visium_sdata.images[f'{image_name}_hires_image'].shape
    box_poly_list = generate_random_boxes(
        image_X= shape_X,
        image_Y = shape_Y
    )

    box_poly_list.append(
        box(
            minx = 0,
            miny = 0,
            maxx = shape_X,
            maxy = shape_Y
        )
    )

    box_poly_df = gpd.GeoDataFrame(geometry=box_poly_list,crs = None)
    box_poly_bounds = box_poly_df.total_bounds.tolist()
    print(f'boxes bounds: {box_poly_bounds}')

    target_bounds = visium_sdata.shapes[image_name].total_bounds.tolist()
    print(f'target_bounds: {target_bounds}')

    # creating dummy landmarks
    #box_landmarks = get_landmarks(box_poly_bounds)
    #target_landmarks = get_landmarks(target_bounds)

    transform = sd.transformations.get_transformation_between_landmarks(
        references_coords = sd.models.ShapesModel.parse(
            np.array([
                [box_poly_bounds[1],box_poly_bounds[0]],
                [box_poly_bounds[3],box_poly_bounds[2]]
            ]),
            geometry = 0,
            radius = 100
        ),
        moving_coords = sd.models.ShapesModel.parse(
            np.array([
                [target_bounds[0],target_bounds[1]],
                [target_bounds[2],target_bounds[3]]
            ]),
            geometry = 0,
            radius = 100
        )
    )
    print(transform)

    visium_sdata.shapes['Random_Boxes'] = sd.models.ShapesModel.parse(
        box_poly_df
    )
    print(visium_sdata)

    other_transform = sd.transformations.align_elements_using_landmarks(
        references_coords = sd.models.ShapesModel.parse(
            np.array([
                [box_poly_bounds[1],box_poly_bounds[0]],
                [box_poly_bounds[3],box_poly_bounds[2]],
                [box_poly_bounds[1],box_poly_bounds[2]],
                [box_poly_bounds[3],box_poly_bounds[0]]
            ]),
            geometry = 0,
            radius = 100
        ),
        moving_coords = sd.models.ShapesModel.parse(
            np.array([
                [target_bounds[0],target_bounds[1]],
                [target_bounds[2],target_bounds[3]],
                [target_bounds[0],target_bounds[3]],
                [target_bounds[2],target_bounds[1]]
            ]),
            geometry = 0,
            radius = 100
        ),
        reference_element = visium_sdata.shapes['Random_Boxes'],
        moving_element = visium_sdata[f'{image_name}_hires_image'],
        reference_coordinate_system='global',
        moving_coordinate_system=image_name,
        new_coordinate_system='aligned'
    )
    print(other_transform)

    postpone_transformation(
        sdata = visium_sdata,
        transformation = other_transform,
        source_coordinate_system='global',
        target_coordinate_system = 'aligned'
    )
    
    print(visium_sdata)
    visium_sdata.shapes['Random_Boxes'] = visium_sdata.shapes['Random_Boxes'].iloc[0:-2,:]

    visium_sdata.pl.render_images().pl.render_shapes().pl.show('aligned')






if __name__=='__main__':
    main()















