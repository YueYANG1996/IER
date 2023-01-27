"""
Contains the code to align wikihow steps with howto100m videos.
To run this scripts, you need to first pre-compute the embeddings
using the MIL-NCE model (s3dg.py).
"""
import torch as th
import pickle
from tqdm import tqdm

cuda = th.device('cuda:0')

def main():
    # load training videos
    meta_data = pickle.load(open("../generalization/generalization_meta.p", "rb"))
    training_goals = [goal for goal in meta_data if meta_data[goal]['split'] == 'train']
    train_videos = []
    for goal in training_goals:
        train_videos += meta_data[goal]["train_videos"] + meta_data[goal]["test_videos"]

    # load text embedding
    # batch size 512
    # every batch [512, 512]
    wiki_ind2steps = pickle.load(open("wiki_dict.p", "rb"))
    all_wikisteps = list(wiki_ind2steps.keys())
    text_embeddings = []
    n_iter = len(all_wikisteps) // 512
    for i in tqdm(range(n_iter + 1)):
        txt_emb = pickle.load(open('text_embedding_wiki/' + str(i) + '.p', 'rb')) # to gpu
        text_embeddings.append(txt_emb) #1105591 * 512
    text_embedding = th.vstack(text_embeddings).to(cuda)

    # run the alignment by multiply the text and image features
    batch_size = 16
    n_iter = len(train_videos) // batch_size
    for i in tqdm(range(n_iter)):
        batch_videos = train_videos[i*batch_size : (i+1)*batch_size]
        video2cluster = {}
        slice_info = []
        for video in batch_videos:
            try:
                cluster = pickle.load(open("clip_clusters/" + video + ".p", "rb"))
            except:
                cluster = pickle.load(open("clip_clusters/" + video, "rb"))
            video2cluster[video] = cluster
            slice_info.append(cluster.shape[0])
        video_embs = th.vstack([pickle.load(open("join_vector_mean/" + video + '.p', 'rb'))[video2cluster[video]] for video in batch_videos]).to(cuda)
        all_scores = th.matmul(video_embs, text_embedding.T)
        step_inds = th.argsort(all_scores, axis = 1, descending=True)[:,:100]
        result = []
        for i, indices in enumerate(step_inds):
            curr_result = []
            for ind in indices:
                curr_result.append((int(ind), float(all_scores[i, ind])))
            result.append(curr_result)
        start_ind = 0
        for i in range(batch_size):
            end_ind = start_ind + slice_info[i]
            pickle.dump(result[start_ind:end_ind], open("retrieval_results/" + batch_videos[i] + '.p', "wb"))
            start_ind = end_ind
        del video_embs
        del all_scores
        del step_inds
    
if __name__ == "__main__":
    main()