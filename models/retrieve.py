import sys
import copy
import pickle
import numpy as np
import torch as th
from s3dg import S3D
from tqdm import tqdm
from edit import edit_schema_howto, edit_schema

cuda = th.device('cuda:0')
# Instantiate the model
net = S3D('../../howtovideo/alignment/s3d_dict.npy', 512)
# Load the model weights
net.load_state_dict(th.load('../../howtovideo/alignment/s3d_howto100m.pth'))
net = net.eval()

refer_k = 1 # number of goals for editing
# editing modules
replacement = True
deletion = True
mask = True

lamb = 0.6 # weight to ensemble global and local matching scores

def compute_goal_score(goal2videos, videos, video2embedding, cuda):
    all_goals = list(goal2videos.keys())
    text_output = net.text_module(all_goals)
    text_embedding = text_output['text_embedding'].to(cuda)
    scores = []
    for video in videos:
        try:
            video_embed = video2embedding[video].to(cuda)
        except:
            cluster = pickle.load(open("../alignment/clip_clusters/" + video + ".p", "rb"))
            video_embed = pickle.load(open("../alignment/join_vector_mean/" + video + '.p', 'rb'))[cluster].to(cuda)
        curr_scores = th.matmul(text_embedding, video_embed.t())
        scores.append(th.mean(curr_scores, dim = 1).tolist())
    scores = np.array(scores).T
    result = {}
    for i in range(len(all_goals)):
        result[all_goals[i]] = scores[i]
    return result


def compute_step_score(goal2steps, videos, video2embedding, cuda):
    all_steps = []
    all_slices = []
    all_goals = list(goal2steps.keys())
    for goal, steps in goal2steps.items():
        all_steps += steps
        all_slices.append(len(steps))
    text_output = net.text_module(all_steps)
    text_embedding = text_output['text_embedding'].to(cuda)
    result = []
    for video in videos:
        try:
            video_embed = video2embedding[video].to(cuda)
        except:
            cluster = pickle.load(open("../alignment/clip_clusters/" + video + ".p", "rb"))
            video_embed = pickle.load(open("../alignment/join_vector_mean/" + video + '.p', 'rb'))[cluster].to(cuda)
        curr_scores = th.matmul(text_embedding, video_embed.t())
        list_of_scores = th.split(curr_scores, all_slices)
        curr_result = []
        for i in range(len(list_of_scores)):
            scores = list_of_scores[i]
            curr_result.append(th.mean(th.max(scores, axis = 1).values).tolist())
        result.append(curr_result)
    result = np.array(result).T
    new_result = {}
    for i in range(len(all_slices)):
        new_result[all_goals[i]] = result[i]
    return new_result  


def get_recall(goal2videos, goal2scores, video2ind, K):
    results = []
    rank_results = []
    goal2rank = {}
    for goal, raw_scores in goal2scores.items():
        videos = goal2videos[goal]
        candidates = [video2ind[video] for video in videos]
        rank = np.argsort(raw_scores)[::-1][:K]
        print(rank)
        correct = 0
        for index in rank:
            if index in candidates:
                correct += 1
        if K == 1:
            results.append(correct / 1)
        else:
            results.append(correct / len(candidates))
        current_rank = []
        for i, ind in enumerate(candidates):
            R = int(np.argwhere(np.argsort(raw_scores)[::-1] == ind)) + 1
            current_rank.append(R)
        rank_results.append(current_rank)
        goal2rank[goal] = current_rank
    
    all_rank = []
    for rank in rank_results:
        all_rank += rank
    med_r = np.median(all_rank)
    mean_r = np.mean(all_rank)
    rr = []
    for rank in rank_results:
        for r in rank:
            rr.append(1/r)
    mrr = np.average(rr)
    return np.average(results), mean_r, med_r, mrr, goal2rank


def howtogen():
    meta_data = pickle.load(open("../generalization/generalization_meta.p", "rb"))
    split = pickle.load(open("../data/howto_gen_goal_split.p", "wb"))

    train_goals = split["train goals"]
    test_goals = split["test goals"]
    
    goal2videos_train = {}
    for goal, data in meta_data.items():
        if data['split'] == 'train':
            goal2videos_train[goal] = data['test_videos']

    train_videos = []
    for goal in goal2videos_train.keys():
        train_videos += goal2videos_train[goal]

    ind2video_train = {}
    video2ind_train = {}
    for i in range(len(train_videos)):
        ind2video_train[i] = train_videos[i]
        video2ind_train[train_videos[i]] = i

    # negative videos in the retrieval pool
    goal2videos_distract = pickle.load(open("distractor_goal2videos_random.p", "rb"))
    distract_videos = []
    for goal, videos in goal2videos_distract.items():
        distract_videos += videos

    video2embedding = {}
    for video in tqdm(distract_videos):
        cluster = pickle.load(open("../alignment/clip_clusters/" + video + ".p", "rb"))
        video_embed = pickle.load(open("../alignment/join_vector_mean/" + video + '.p', 'rb'))[cluster]
        video2embedding[video] = video_embed
    
    data_induction = {k:v for k, v in pickle.load(open("../data/goal2steps_clustered_full.p", "rb")).items() if k in train_goals}
    testgoal2traingoals = pickle.load(open("testgoal2traingoals_sentence_max.p", "rb"))
    schema_resource = data_induction
    ranks = []

    for goal in tqdm(test_goals):
        list_schema_steps, list_hops = edit_schema_howto(goal, testgoal2traingoals, meta_data, schema_resource, refer_k, replacement, deletion, mask)
        data_generalization = {i:[goal] + list_schema_steps[i] for i in range(refer_k)}
        
        gen_goal2videos_test = copy.deepcopy(goal2videos_distract)
        gen_goal2videos_test[goal] = meta_data[goal]['test_videos']
        test_videos = []
        for goal in gen_goal2videos_test.keys():
            test_videos += gen_goal2videos_test[goal]

        ind2video_test = {}
        video2ind_test = {}
        for i in range(len(test_videos)):
            ind2video_test[i] = test_videos[i]
            video2ind_test[test_videos[i]] = i

        test_goal_score = compute_goal_score({goal:[]}, test_videos, video2embedding, cuda)
        test_step_score = compute_step_score(data_generalization, test_videos, video2embedding, cuda)
        
        step_scores = list_hops[0] * test_step_score[0]
        for i in range(1, refer_k):
            step_scores += list_hops[i] * np.array(test_step_score[i])
        step_scores = step_scores / refer_k
        test_step_score = {goal: step_scores}

        combined_scores = {}
        for goal in test_step_score.keys():
            combined_scores[goal] = (1 - lamb) * np.array(test_goal_score[goal]) + lamb * np.array(test_step_score[goal])
        
        goal2rank_step = get_recall(gen_goal2videos_test, combined_scores, video2ind_test, 1)[-1]
        ranks.append(goal2rank_step[goal])
    
    def compute_performance(ranks):
        K2recalls = {}
        for k in [1, 5, 10, 25, 50]:
            recalls = []
            for rank in ranks:
                correct = 0
                for r in rank:
                    if r <= k:
                        correct += 1
                if k == 1:
                    recalls.append(correct / 1)
                else:
                    recalls.append(correct / len(rank))
            K2recalls[k] = np.mean(recalls)
        
        all_ranks = []
        for rank in ranks:
            all_ranks += rank
        
        med_r = np.median(all_ranks)
        mean_r = np.mean(all_ranks)
        
        rr = []
        for rank in ranks:
            for r in rank:
                rr.append(1/r)
        mrr = np.average(rr)
        return K2recalls, med_r, mean_r, mrr

    print(compute_performance(ranks))


def coin():
    coin_goal2test_videos = pickle.load(open("../data/coin/coin_goal2test_videos.p", "rb"))
    all_goals = list(coin_goal2test_videos.keys())

    test_videos = []
    for _, videos in coin_goal2test_videos.items():
        test_videos += videos

    ind2video_test = {}
    video2ind_test = {}
    for i in range(len(test_videos)):
        ind2video_test[i] = test_videos[i]
        video2ind_test[test_videos[i]] = i

    video2embedding = {}
    for video in tqdm(test_videos):
        video_embed = pickle.load(open("../data/coin/joint_vector_mean/" + video + '.p', "rb"))
        video2embedding[video] = video_embed

    # global scores
    test_goal_score = compute_goal_score(all_goals, test_videos, video2embedding, cuda)

    howto_goal2steps_induction = pickle.load(open("../data/goal2steps_clustered_full.p", "rb"))
    howto_goal2verb_noun = pickle.load(open("../data/goal2verb_noun.p", "rb"))
    coin_goal2noun = pickle.load(open("../data/coin/coin_goal2noun.p", "rb"))
    coin_goal2howto_goal = pickle.load(open("../data/coin/coin_goal2howto_goal_max.p", "rb"))

    for k in range(refer_k):
        goal2steps_induction = {}
        goal2hops = {}
        for goal in tqdm(all_goals):
            schema_steps, hops = edit_schema(goal, coin_goal2noun, coin_goal2howto_goal, k, howto_goal2steps_induction, howto_goal2verb_noun, replacement, deletion, mask)
            goal2steps_induction[goal] = [goal] + schema_steps
            goal2hops[goal] = hops
        current_step_score = compute_step_score(goal2steps_induction, test_videos, cuda)
        if k == 0:
            test_step_score = {}
            for goal in all_goals:
                test_step_score[goal] = np.array(current_step_score[goal]) * goal2hops[goal]
        else:
            for goal in all_goals:
                test_step_score[goal] += np.array(current_step_score[goal]) * goal2hops[goal]

    combined_scores = {}
    for goal in test_step_score.keys():
        combined_scores[goal] = (1 - lamb) * np.array(test_goal_score[goal]) + lamb * (1/refer_k) * np.array(test_step_score[goal])
    
    for k in [1, 5, 10, 25, 50]:
        recall, mean_r, med_r, mrr, _ = get_recall(coin_goal2test_videos, combined_scores, video2ind_test, k)
        print(recall, mean_r, med_r, mrr)


def youcook2():
    goal2videos = pickle.load(open("../data/youcook2/youcook2_goal2videos.p", "rb"))
    all_goals = list(goal2videos.keys())

    test_videos = []
    for _, videos in goal2videos.items():
        test_videos += videos

    ind2video_test = {}
    video2ind_test = {}
    for i in range(len(test_videos)):
        ind2video_test[i] = test_videos[i]
        video2ind_test[test_videos[i]] = i

    howto_goal2steps_induction = pickle.load(open("../data/goal2steps_clustered_full.p", "rb"))
    howto_goal2verb_noun = pickle.load(open("../data/goal2verb_noun.p", "rb"))
    youcook_goal2noun = pickle.load(open("../data/youcook2/youcook_goal2noun.p", "rb"))
    youcook_goal2howto_goal = pickle.load(open("../data/youcook2/youcook_goal2howto_goal_max.p", "rb"))

    for k in range(refer_k):
        goal2steps_induction = {}
        goal2hops = {}
        for goal in tqdm(all_goals):
            schema_steps, hops = edit_schema(goal, youcook_goal2noun, youcook_goal2howto_goal, k, howto_goal2steps_induction, howto_goal2verb_noun, replacement, deletion, mask)
            goal2steps_induction[goal] = [goal] + schema_steps
            goal2hops[goal] = hops

        current_step_score = compute_step_score(goal2steps_induction, test_videos, cuda)
        if k == 0:
            test_step_score = {}
            for goal in all_goals:
                test_step_score[goal] = np.array(current_step_score[goal]) * goal2hops[goal]
        else:
            for goal in all_goals:
                test_step_score[goal] += np.array(current_step_score[goal]) * goal2hops[goal]

    test_goal_score = compute_goal_score(all_goals, test_videos, cuda)

    lamb = 0.6
    combined_scores = {}
    for goal in test_step_score.keys():
        if goal in all_goals:
            combined_scores[goal] = (1 - lamb) * np.array(test_goal_score[goal]) + lamb * (1/refer_k) * np.array(test_step_score[goal])
    
    for k in [1, 5, 10, 25, 50]:
        recall, mean_r, med_r, mrr, _ = get_recall(goal2videos, combined_scores, video2ind_test, k)
        print(recall, mean_r, med_r, mrr)


def main(dataset):

    if dataset == "howtogen":
        howtogen()
    elif dataset == "coin":
        coin()
    elif dataset == "youcook2":
        youcook2()
    else:
        print("dataset not supported")


if __name__ == "__main__":
    dataset = sys.argv[1]
    main(dataset)