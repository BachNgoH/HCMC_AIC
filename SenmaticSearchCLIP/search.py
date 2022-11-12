from feature_extractor import compute_clip_feature
import numpy as np
import faiss
import pandas as pd
import clip
import torch
# pip install googletrans==3.1.0a0
# pip install langdetect
# !pip install translate

from langdetect import detect
import googletrans
import translate

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class Translation():
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        return text.lower()

    def __call__(self, text):
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text

# translate text query and extract feature text query 
def get_feature_vector(text_query):
    #translator = Translation()

    #if detect(text_query) == 'vi':
    #    text = translator(text_query)

    text = clip.tokenize([text_query]).to(device)  
    text_features = model.encode_text(text).cpu().detach().numpy().astype(np.float32)
    print(text_features.shape)
    return text_features

def get_image_feature_vector(photo_id):
    photo_ids = pd.read_csv("./util/photo_ids_1M.csv")
    photo_ids = list(photo_ids['photo_id'])

    photo_idx = photo_ids.index(photo_id)
    photo_features = np.load("./util/features_1M.npy")

    photo_feature = photo_features[photo_idx].astype(np.float32)

    photo_feature = np.expand_dims(photo_feature, axis=0)
    return photo_feature

# creat file faiss bin
def create_index_vector(photo_features):
    # build the index by dimension and add vectors to the index
    index = faiss.IndexFlatL2(512)
    fe = photo_features.reshape(photo_features.shape[0], -1).astype('float32')
    index.add(fe)
    # write index file to disk
    faiss.write_index(index, 'faiss_L2.bin')
    print('done!!')


def search_vector(query, photo_index, topk, mode="text"):

    if(mode == "photo"):
        features_vector_search = get_image_feature_vector(query)
    else:
        features_vector_search = get_feature_vector(query)
    
    # print(features_vector_search.shape)
    # search image by the feature vector
    f_dist, f_ids = photo_index.search(features_vector_search, topk)  # actual search

    # print(f_dists[0][1:]) #score
    # print(f_ids) # id

    photo_index.reset()
    return f_ids[0] # lay id 


if __name__ == '__main__':
    
    # create file bin
    # create_index_vector(photo_features)

    # TEST
    # text_query = 'Người nghệ nhân đang tô màu cho chiếc mặt nạ một cách tỉ mỉ. Xung quanh ông là rất nhiều những chiếc mặt nạ. Người nghệ nhân đi đôi dép tổ ong rất giản dị. Sau đó là hình ảnh quay cận những chiếc mặt nạ. Loại mặt nạ này được gọi là mặt nạ giấy bồi Trung thu.'
    text_query = 'The artist is painting the mask meticulously. Around him were many masks. The artist wears very simple honeycomb sandals. Then there is a close-up image of the masks. This type of mask is called the Mid-Autumn Festival paper mask.'

    photo_features = np.load("./features_1M.npy")
    photo_index = faiss.read_index("./faiss_L2.bin")
    k = 5

    photo_ids = pd.read_csv("./photo_ids_1M.csv")
    photo_ids = list(photo_ids['photo_id'])
    ids = search_vector(text_query, photo_index, k)
    
    for i in ids:
        print(photo_ids[i])
