def get_features(tiles):
    import torch 

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()

    # forces values to 0-255 int values. 
    def preprocess_tile(tile):
        import torchvision.transforms as transforms
        import numpy as np
        from PIL import Image

        tile = tile / np.max(tile) # normalizes from 0-1 because I've been using different image types with different intensity ranges so this puts everyting on equal starting point 
        tile_pil = Image.fromarray((tile * 255).astype(np.uint8)) # now I make it standard 0-255. 
        tile_pil = tile_pil.convert("RGB")

        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # to match imagnet training. has not caused issues so far. 
        ])

        return transform(tile_pil)

    input_tensor = torch.stack([preprocess_tile(tile) for tile in tiles])

    with torch.no_grad():
        features = model(input_tensor)

    features = features.squeeze().cpu().numpy()

    return features

def get_distances(features, index:int=None, n_closest:int=5):
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    # note: the n_closest determines how many closest images will be returned. this is important as it dtermines what goes into the distances, closest_indices arrays, and influences what hte comparison_plot will look like. 

    distances = euclidean_distances(features)
    
    if index is not None:
        closest_indices = np.argsort(distances[index])[1:n_closest+1]
        return distances, closest_indices

    else: 
        return distances 


def comparision_plot(tiles, features, index:int, closest_indices, plot_path:str=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import smplotlib 

    n = len(closest_indices)

    fig, axs = plt.subplots(1, 1+n, figsize=((1+n)*3, 3))

    axs[0].imshow(tiles[index], cmap='Greys_r')
    axs[0].set(title='Reference')

    for i, idx in enumerate(closest_indices): 
        axs[i+1].imshow(tiles[idx], cmap='Greys_r')

        distance_str = str(np.round(np.linalg.norm(features[index]-features[idx]), 2))
        distance_str = distance_str.split(".")[0]+'.'+distance_str.split(".")[-1][0:1]
        
        axs[i+1].set(title=f'Distance: {distance_str}')
        axs[i+1].set_xticklabels([])
        axs[i+1].set_yticklabels([])

    if plot_path is not None: 
        plt.savefig(plot_path)

    plt.show()

images = None # list of 224x224 numpy arrays corresponding to the images we want to study.  
query_index = None # index of the image in that array that we want to use as reference to draw comparisons from. my reccomendation is you make the array to be [reference_image, ... everything else you want to draw comparisons from ...]

features = get_features(images)

distances, closest_indices = get_distances(features, index=query_index)

# now, if you want you can plot some of them 

