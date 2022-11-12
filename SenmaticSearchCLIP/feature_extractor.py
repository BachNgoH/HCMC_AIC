# pip install git+https://github.com/openai/CLIP.git

import clip
import torch
from PIL import Image
import pandas as pd 

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function that computes the feature vectors for a batch of images
def compute_clip_feature(img_path):
    
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feature = model.encode_image(image)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return img_feature.cpu().numpy()

if __name__ == '__main__':
    # test 
    # path = 'SenmaticSearchCLIP/static/data/KeyFramesC00_V00/C00_V0015/010448.jpg'

    # fe = compute_clip_feature(path)
    # print(fe)

    photo_ids = pd.read_csv("SenmaticSearchCLIP/photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])
    print(photo_ids.index(1))

