def get_images(data_dir, img_type:str='pan', cutout_dir:str='cutout_images'): 
    import os
    from PIL import Image
    import numpy as np

    os.makedirs(cutout_dir, exist_ok=True)

    image_cutouts = []
    image_names = []

    for file in os.listdir(data_dir): 
        if img_type in file and "png" in file: 
            file_path = os.path.join(data_dir, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width >= 244 and height >= 244:
                        cutout = img.crop((0, 0, 244, 244))
                        cutout_array = np.array(cutout)
                        image_cutouts.append(cutout_array)
                        
                        image_name = file.replace('.png','')
                        image_names.append(image_name)

                        # Save cutout image to directory
                        cutout.save(os.path.join(cutout_dir, f"{image_name}_cutout.png"))

            except Exception as e:
                print(f"Error for {file_path}: {e}")

    image_cutouts = np.array(image_cutouts)
    image_names = np.array(image_names)

    indices = np.random.permutation(len(image_cutouts))
    image_cutouts = image_cutouts[indices]
    image_names = image_names[indices]

    return image_cutouts, image_names

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

def comparision_plot(tiles, features, index:int, closest_indices, plot_dir:str=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os 

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

    if plot_dir is not None: 
        plt.savefig(os.path.join(plot_dir, f'comparison_i={index}.png'))

    plt.show()

images, image_names = get_images(data_dir='/burg/home/tjk2147/src/GitHub/transformer-knn/data') 

features = get_features(images)

import pickle 
# save because matplotlib is being weird
features_dict = {name: feature for name, feature in zip(image_names, features)}
with open('features_dict.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

plot_dir = '/burg/home/tjk2147/src/GitHub/transformer-knn/code/monthly/getdinolatent/plots'

query_index = 5
distances, closest_indices = get_distances(features, index=query_index)
comparision_plot(images, features, index=query_index, closest_indices=5, plot_dir=plot_dir)

query_index = 3
distances, closest_indices = get_distances(features, index=query_index)
comparision_plot(images, features, index=query_index, closest_indices=5, plot_dir=plot_dir)

query_index = 1
distances, closest_indices = get_distances(features, index=query_index)
comparision_plot(images, features, index=query_index, closest_indices=5, plot_dir=plot_dir)