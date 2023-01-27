"""
Contains the code to induce schemata (set of steps) for 
howto100m tasks. You need to first run align.py to get 
the alignment results between wikihow steps and videos.
"""
import csv
import pickle
from tqdm import tqdm
import numpy as np
import torch as th

def get_k_labels(video, k):
    alignments = pickle.load(open("retrieval_results/" + video + '.p', "rb"))
    results = []
    for alignment in alignments:
        results.append(alignment[:k])
    return results


def sort_steps(goals, taskid2videos, goal2taskid, ind2step):
    goal2steps_sorted = {}
    for goal in tqdm(goals):
        videos = taskid2videos[goal2taskid[goal]]
        step2score = {}
        k = 100
        count = 0
        for video in videos:
            filtered_step_labels = get_k_labels(video, 30)
            try:
                step2score_current = {}
                for clip_info in filtered_step_labels:
                    for step, score in clip_info:
                        if step not in step2score_current:
                            step2score_current[step] = [score]
                        else:
                            step2score_current[step].append(score)
                for step, scores in step2score_current.items():
                    if step not in step2score:
                        step2score[step] = np.mean(scores)
                    else:
                        step2score[step] += np.mean(scores)
                count += 1
            except:
                continue
        sorted_steps = sorted(step2score.items(), key=lambda item: item[1], reverse=True)
        scores = [score for _, score in sorted_steps]
        top_steps = [(ind2step[ind], score / count) for ind, score in sorted(step2score.items(), key=lambda item: item[1], reverse=True)[:k]]
        goal2steps_sorted[goal] = top_steps
    return goal2steps_sorted


def cluster_steps(goal2steps_sorted):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    embedder = SentenceTransformer('all-mpnet-base-v2', device = th.device('cuda:0'))

    goal2steps_clustered = {}
    for goal, step_w_score in tqdm(goal2steps_sorted.items()):
        corpus = [step for step, _ in step_w_score]
        corpus_embeddings = embedder.encode(corpus)

        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # Perform kmean clustering
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []

            clustered_sentences[cluster_id].append(corpus[sentence_id])
        
        clustered_steps = []
        for cluster_id, steps in clustered_sentences.items():
            if len(steps) > 1:
                clustered_steps.append(steps[0])
        goal2steps_clustered[goal] = clustered_steps

    return goal2steps_clustered


def main():
    meta_data = pickle.load(open("../data/generalization_meta.p", "rb"))
    ind2step = pickle.load(open("../models/wiki_dict.p", "rb"))
    taskid2videos = pickle.load(open("../data/taskid2videos.p", "rb"))

    train_goals = []
    for goal, data in meta_data.items():
        if data['split'] == 'train':
            train_goals.append(goal)

    taskid2goal = {}
    with open('../data/task_ids.csv', newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        for row in reader:
            taskid2goal[row[0]] = row[1]
            
    goal2taskid = {goal: taskid for taskid, goal in taskid2goal.items()}

    all_howtogoals = []
    for taskid, videos in taskid2videos.items():
        if len(videos) >= 20:
            all_howtogoals.append(taskid2goal[taskid])

    # sort steps by alignment scores
    goal2steps_sorted = sort_steps(all_howtogoals, taskid2videos, goal2taskid, ind2step)

    # cluster steps to remove parapharses
    goal2steps_clustered = cluster_steps(goal2steps_sorted)

    # save the induced schemata which is also provided in the repo
    pickle.dump(goal2steps_clustered, open("goal2steps_clustered_full.p", "wb"))


if __name__ == "__main__":
    main()