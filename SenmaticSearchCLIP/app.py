from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import clip
import torch
from flask_cors import CORS
import csv
from PIL import Image
from sentence_transformers import SentenceTransformer, util

UPLOAD_FOLDER = "./tmp"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

ocr_model = SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

@app.route('/api', methods = ['GET'])
def get_frame_from_query():
    query = request.args.get("q")
    page = int(request.args.get("page"))
    photo_id = request.args.get("photoId")
    
    photo_features = np.load("./features.npy")
    photo_ids = pd.read_csv("./photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])


    if (query and not photo_id):
        with torch.no_grad():
            text_encoded = model.encode_text(clip.tokenize(query).to(device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        
        text_features = text_encoded.cpu().numpy()
        similarities = list((text_features @ photo_features.T).squeeze(0))
        best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)

    elif (photo_id and not query):
        photo_idx = photo_ids.index(photo_id)

        photo_feat = photo_features[photo_idx]

        similarities = list(photo_feat @ photo_features.T)
        best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)
    # else:
    #     photo_idx = photo_ids.index(photo_id)
    #     photo_feat = photo_features[photo_idx]



    
    results = []
    if ((page - 1) * 100 > len(best_photos)):
        return jsonify({'data': "NO MORE DATA"})
    for i in range((page - 1) * 100, page*100):
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        results.append(photo_id)

    results = list(dict.fromkeys(results))

    return jsonify({'data': results})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/photo', methods=["POST"])
def search_photo():
    
    if request.method == 'POST':
        photo_features = np.load("./features.npy")
        photo_ids = pd.read_csv("./photo_ids.csv")
        photo_ids = list(photo_ids['photo_id'])
        print("successfull")
        print(request.files)
        if 'file' not in request.files:
            print('No file part')
            return
        file = request.files['file']
        print("on here")
        if file.filename == '':
            print('No selected file')
            return 
        print("on here")
        if file and allowed_file(file.filename):
            print("file get successfull")
            image = Image.open(file)            

            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_encoded = model.encode_image(image_input)
                image_encoded /= image_encoded.norm(dim=-1, keepdim=True)
            image_encoded = image_encoded.cpu().numpy()
            # Compute the similarity between the descrption and each photo using the Cosine similarity
            similarities = list((image_encoded @ photo_features.T).squeeze(0))

            # Sort the photos by their similarity score
            best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)
            
            
            results = []
            # if ((page - 1) * 100 > len(best_photos)):
            #     return jsonify({'data': "NO MORE DATA"})
            for i in range(100):
                idx = best_photos[i][1]
                photo_id = photo_ids[idx]
                results.append(photo_id)

                results = list(dict.fromkeys(results))

            return jsonify({'data': results})
           

    # photo_idx = photo_ids.index(photo_id)

    # photo_feat = photo_features[photo_idx]

    # similarities = list(photo_feat @ photo_features.T)
    # best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)

    # results = []
    # for i in range(100):
    #     idx = best_photos[i][1]
    #     photo_id = photo_ids[idx]
    #     results.append(photo_id)
    # results = list(dict.fromkeys(results))
    # return jsonify({'data': results})

@app.route('/video/<string:photo_id>')
def get_frame_video(photo_id):
    photo_ids = pd.read_csv("./photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])

    results = [item for item in photo_ids if item.startswith(photo_id[:9])]
    results = list(dict.fromkeys(results))

    return jsonify({'data': results})

# def vector_search(query, model, embeddings, num_results=10):
#     """
#     Args:
#         query (str): User query that should be more than a sentence long.
#         model (sentence_transformers.SentenceTransformer.SentenceTransformer)
#         index (`numpy.ndarray`): FAISS index that needs to be deserialized.
#         num_results (int): Number of results to return.
    
#     Returns:
#         D (:obj:`numpy.array` of `float`): Distance between results and query.
#         I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
#     """
#     vector = model.encode(list(query))
#     cos_scores = util.cos_sim(np.array(vector), embeddings)[0]
#     D, I = torch.topk(cos_scores, k=num_results)
#     return D, I

# @app.route("/ocr", methods=["GET"])
# def get_ocr_video():
#     query = request.args.get("q")
#     # page = int(request.args.get("page"))

#     # Step 1: Change data type
#     embeddings = np.load("./embeddings.npy")

#     keys = np.load("./keys.npy")

#     # Retrieve the 10 nearest neighbours
#     D, I = vector_search(query, ocr_model, embeddings, 100)

#     results = [keys[idx][:-4] for idx in I]
#     return jsonify({"data": results})    

@app.route("/submission", methods = ['POST'])
def get_submission():

    content = request.get_json()
    submission_list = content["data"]

    first_submission = submission_list[0]
    
    photo_ids = pd.read_csv("./photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])
    photo_features = np.load("./features.npy")

    photo_id = f"{first_submission['video'][:-4]}-{first_submission['frame']}"

    photo_idx = photo_ids.index(photo_id)

    photo_feat = photo_features[photo_idx]

    similarities = list(photo_feat @ photo_features.T)
    best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)
   
    i = 0
    while(len(submission_list) <  100):
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        sub_dict = {"video": f"{photo_id[:9]}.mp4", "frame": photo_id[10:],};
        if (sub_dict not in submission_list):
            submission_list.append(sub_dict)
        i +=1
    
    new_submission = []
    for submission in submission_list:
        video = submission['video']
        frame = submission['frame']
        
        frame_dict = {}
        with open(f"./keyframe_p/{video[:9]}.csv") as f:
            reader = csv.reader(f)
            # print(list(reader))
            for item in list(reader):
                frame_dict[item[0]] = item[1]

            new_frame = frame_dict[f'{frame}.jpg']

            new_submission.append({
                "video": video,
                "frame": new_frame,
            })
    print(new_submission, "NEW SUB")
    return jsonify({"data": new_submission})

if __name__ == '__main__':

    app.run(debug=True)

