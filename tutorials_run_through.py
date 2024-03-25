"""

Going through tutorials/examples in squidpy


"""

import os
import sys

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt

from math import floor

# Function to cluster new features present in adata
def cluster_features(features: pd.DataFrame, like=None) -> pd.Series:
    """
    Calculate leiden clustering of features.

    Specify filter of features using `like`.
    """
    # filter features
    if like is not None:
        features = features.filter(like=like)
    # create temporary adata to calculate the clustering
    adata = ad.AnnData(features)
    # important - feature values are not scaled, so need to scale them before PCA
    sc.pp.scale(adata)
    # calculate leiden clustering
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)

    return adata.obs["leiden"]

def generate_random_boxes(image_Y,image_X,n_boxes=200,max_size=1000):
    """
    Generate random box masks for a given image. Outputs labeled mask that is the same spatial dimensions as the original image (image_Y,image_X)
    
    Does not check if boxes overlap
    """
    box_mask = np.zeros((image_Y,image_X))

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

        # Creating labeled mask
        box_mask[top:bottom, left:right] = b+1
    
    return box_mask

def box_count_fn(arr, n_box):
    """
    Input array from image for feature extraction.
    Returns the area that each labeled box occupies for a given spot
    """
    #plt.imshow(np.squeeze(arr))
    #plt.show()
    box_intersect_list = []
    for b in range(1,n_box):
        box_intersect_list.append(np.sum(arr==b))
    
    return box_intersect_list


def main():
    

    # Analyzing Visium H&E data:
    # https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_visium_hne.html

    # Loading pre-processed dataset
    img = sq.datasets.visium_hne_image()
    adata = sq.datasets.visium_hne_adata()

    # Plotting named clusters on spots overlaid on mouse brain
    sq.pl.spatial_scatter(adata, color = 'cluster',save = './visium_spatial_scatter.png')
    """
    # Calculating image features at different scales of resolution
    for scale in [4.0]:
        #feature_name = f"features_summary_scale{scale}"
        feature_name = 'custom_mean_channels'
        sq.im.calculate_image_features(
            adata,
            img.compute(),
            #features=['summary','texture'],
            features = 'custom',
            features_kwargs = {'custom':{'func':np.mean,'axis': 0}},
            key_added=feature_name,
            #n_jobs=4,
            scale=scale,
        )

    # Features stored in adata.obsm
    # Combining multiple scales of features into one dataframe
    
    adata.obsm["features"] = pd.concat(
        [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f],
        axis="columns",
    )
    
    # Fixing duplicate feature names
    adata.obsm["features"].columns = ad.utils.make_index_unique(
        adata.obsm["features"].columns
    )

    print(f'features calculated: {adata.obsm["features"].columns.tolist()}')
    
    # calculate feature clusters
    adata.obs["features_cluster"] = cluster_features(adata.obsm["custom_mean_channels"])

    # compare feature and gene clusters
    # Groups can be specified so that only the spots with that label are included
    sq.pl.spatial_scatter(
        adata,
        groups = ['1','2','3'],
        color=["features_cluster"],
        img = True,
        save='./cluster_features.png'
    )

    """
    # Adding custom shape "segmentations" to the image
    print(img.shape)
    image_Y, image_X = img.shape[0], img.shape[1]

    # Mask containing labels for randomly placed boxes
    rando_boxes = generate_random_boxes(image_Y, image_X)
    print('Box mask created')
    print(f'{len(np.unique(rando_boxes))} boxes added')

    # Creating an image container for this annotation
    box_img = sq.im.ImageContainer(
        img = np.int32(np.repeat(rando_boxes[:,:,None],repeats=3,axis=-1)),
        layer = 'boxes'
    )
    box_img['boxes'].attrs['segmentation'] = True
    
    img.add_img(
        img = box_img,
        layer = 'boxes'
    )

    print(img)

    # This one doesn't show up with small number of boxes
    #box_img.show('boxes',save='./boxes.png')
    #img.show('image',segmentation_layer='boxes',segmentation_alpha=0.8,save='./test_boxes.png')

    # This calculates the number of boxes that each spot intersects with
    sq.im.calculate_image_features(
        adata,
        img.compute(),
        layer= 'image',
        features='segmentation',
        features_kwargs = {
            'segmentation': {
                'label_layer': 'boxes',
                'props': ['label']
            }
        },
        key_added = 'box_count',
        mask_circle = True
    )

    sq.im.calculate_image_features(
        adata,
        img.compute(),
        layer='boxes',
        features = 'custom',
        features_kwargs = {
            'custom': {
                'func': box_count_fn,
                'n_box': len(np.unique(rando_boxes))-1
            }
        },
        key_added = 'box_intersect_features',
        mask_circle = True
    )

    # Now doing the reverse (fails)
    sq.im.calculate_image_features(
        adata,
        img.compute(),
        layer = 'boxes',
        features = 'segmentation',
        features_kwargs = {
            'segmentation': {
                'label_layer': 'boxes',
                'props': ['label']
            }
        },
        key_added = 'spot_count'
    )

    """
    squidpy automatically iterates through cropped spot regions no matter what is provided to the "label"

    There is no way to perform the reverse as long as this is hard-coded into calculate_image_features
    
    """

    print(type(adata.obsm['box_count']))
    print(adata.obsm['box_count'].shape)
    print(type(adata.obsm['spot_count']))
    print(adata.obsm['spot_count'].shape)
    adata.obsm['box_count'].to_csv('./box_count_test.csv')
    adata.obsm['box_intersect_features'].to_csv('./box_intersect_features.csv')
    adata.obsm['spot_count'].to_csv('./spot_count_test.csv')








if __name__=='__main__':
    main()



























