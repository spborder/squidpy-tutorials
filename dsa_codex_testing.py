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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale, Sequence, Translation


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

def matplotlib_figure_to_array(fig: Figure) -> np.ndarray:
    """
    Render a Matplotlib figure to a Numpy array.

    Args:
        fig: A figure. The figure's canvas must be a FigureCanvasAgg

    Returns:
        An RGB Numpy array
    """
    # From https://stackoverflow.com/a/7821917
    # matplotlib.pyplot.switch_backend("agg")

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas: FigureCanvasAgg
    fig.canvas.draw()

    # Now we can save it to a Numpy array.
    array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_rgba = array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_rgb = image_rgba[:, :, :3]
    return image_rgb

def create_figure_and_axes(shape: tuple[int, int], resolution: float) -> Axes:
    """
    Create a Matplotlib figure with axes that is fully filled by the plot, no padding/labels/legend.

    Args:
        shape: Shape of the plot area in data units
        resolution: Scale factor for units of the shape to image pixels

    Returns:
        The axes object to plot into
    """
    plt.set_cmap('Greys')
    dpi = 100
    image_size_yx = np.asarray(shape) * resolution
    print(image_size_yx)
    fig = Figure(dpi=dpi, figsize=np.squeeze(np.flip(image_size_yx) / dpi), frameon=False, layout="tight")
    # The Agg backend is required for later accessing FigureCanvasAgg.buffer_rgba()
    fig.canvas.switch_backends(FigureCanvasAgg)
    fig.bbox_inches = fig.get_tightbbox().padded(0)
    ax: Axes = fig.add_axes(rect=(0.0, 0.0, 1.0, 1.0))
    
    return ax



def main():

    # Reading images (Have to load the whole image as a numpy array in order to initialize )
    histology_img = large_image.open('./dsa/15-1 Stitch.svs')
    histo_name = '15-1 Stitch.svs'
    # EC2 itemId
    #histoId = '6461bbe818a9d4c4175edbcd'
    histo_meta = histology_img.getMetadata()
    histo_sizeX, histo_sizeY = histo_meta['sizeX'], histo_meta['sizeY']

    histo_thumbnail,_ = histology_img.getRegion(
        format = large_image.constants.TILE_FORMAT_NUMPY,
        region = {
            'top': 0,
            'left': 0,
            'right': histo_sizeX,
            'bottom': histo_sizeY
        }
    )

    #histo_thumbnail, _ = histology_img.getThumbnail()
    #histo_thumbnail = np.array(Image.open(BytesIO(histo_thumbnail)))
    histo_thumbnail_shape = np.shape(histo_thumbnail)

    # Images are assumed to be (cyx)
    histo_thumbnail = np.moveaxis(histo_thumbnail,source=-1, destination = 0)

    # Repeating for CODEX Image
    codex_img = large_image.open('./dsa/15-1.tif')
    codex_name = '15-1.tif'
    # EC2 itemId
    #codexId = '648b46573e6ae3107da0d990'
    codex_meta = codex_img.getMetadata()
    codex_sizeX, codex_sizeY = codex_meta['sizeX'], codex_meta['sizeY']

    codex_thumbnail, _ = codex_img.getRegion(
        format = large_image.constants.TILE_FORMAT_NUMPY,
        region = {
            'top': 0,
            'left': 0,
            'right': codex_sizeX,
            'bottom': codex_sizeY
        },
        frame = 0
    )

    #codex_thumbnail, _ = codex_img.getThumbnail()
    #codex_thumbnail = np.array(Image.open(BytesIO(codex_thumbnail)))    
    codex_thumbnail_shape = np.shape(codex_thumbnail)
    # Converting thumbnail image to (cyx)
    codex_thumbnail = np.moveaxis(codex_thumbnail, source = -1, destination = 0)
    # Reading smaller size level into memory

    # Scale factors for annotations on thumbnails
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

    # Defining the SpatialData object here puts everything in its own "global" coordinate system
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

    # Coordinate systems with the same name get combined when concatenating, renaming here so that it's unique
    histo_sdata.rename_coordinate_systems({'global':'histo_crs'})
    print(histo_sdata)

    # Repeating for codex data
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

    # Combining two different SpatialData objects into one (shows up as two distinct coordinate systems)
    multi_sdata = sd.concatenate([histo_sdata,codex_sdata])
    print(multi_sdata)

    # Showing the image and landmarks associated with that image
    #multi_sdata.pl.render_images().pl.render_shapes().pl.show('histo_crs')
    #multi_sdata.pl.render_images().pl.render_shapes().pl.show('codex_crs')


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

    # Creates an affine transform sequence to go from one coordinate system to the other
    print(transform)


    # This applies that transform to all the objects in a particular coordinate system
    postpone_transformation(
        sdata = multi_sdata,
        transformation = transform,
        source_coordinate_system = 'codex_crs',
        target_coordinate_system = 'aligned'
    )

    print(multi_sdata)

    # Showing aligned and separate (but transformed) images and landmarks
    #multi_sdata.pl.render_images().pl.render_shapes().pl.show('aligned')
    #multi_sdata.pl.render_images('histo_img').pl.render_shapes().pl.show('aligned')
    #multi_sdata.pl.render_images('codex_img').pl.render_shapes().pl.show('aligned')

    # Writing image array

    # Courtesy of: https://github.com/scverse/spatialdata-plot/issues/204
    bounding_box = sd.get_extent(
        multi_sdata.shapes['codex_landmarks'],
        coordinate_system = 'aligned'
    )
    print(bounding_box)
    query_box = np.array([
        [bounding_box['y'][0], bounding_box['x'][0]],
        [bounding_box['y'][1],bounding_box['x'][1]]
    ])
    print(query_box)
    resolution = 1.0

    cropped_sdata = multi_sdata.query.bounding_box(
        axes = ("y","x"),
        min_coordinate = query_box[0],
        max_coordinate = query_box[1],
        target_coordinate_system = "aligned"
    )

    ax = create_figure_and_axes(shape = tuple(np.diff(query_box,axis=0)), resolution = resolution)
    cropped_sdata.pl.render_images().pl.show(coordinate_systems = "aligned",ax=ax)

    ax.set_ybound(query_box[0][0],query_box[1][0])
    ax.set_xbound(query_box[0][1],query_box[1][1])

    image_array = matplotlib_figure_to_array(ax.figure)
    Image.fromarray(image_array).save('./dsa/test_aligned_image.tif')








if __name__=='__main__':
    main()












