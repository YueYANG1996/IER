# IER
Code for the paper: [Induce, Edit, Retrieve: Language Grounded Multimodal Schema for Instructional Video Retrieval](https://arxiv.org/pdf/2111.09276.pdf)

### Set up environments
We run our experiments using Python 3.9.1. You can install the required packages using:

```
pip install -r requirements.txt
```

## data
* howto_gen_goal_split.p: the train/val/test split of the goals
* generalization_meta.p provides the detailed information of each goal, here is an example:
```
Prepare Fish
verb: Prepare
noun: Fish
test videos: a list of videos for test
train videos: a list of videos for train
wiki_steps: the oracle schema from wikiHow
'split': train/val/test
category: the category of this goal in wikiHow
```
* goal2steps_clustered_full.p the schemata of 21,299 tasks induced by IER

## Evaluate
### Induce
Download the text and video features, and run `align.py` to align wikihow sentences with videos.
Then run `induce.py`, the results will be saved to `data/howto/goal2steps_clustered_full.p`.

### Edit
The code for editing modules is in `models/edit.py`.

### Retrieve
To get the retrieval performance, run:

```
python retrieve.py {DATASET}
```

The `DATASET` can be `howtogen`, `youcook2` or `coin`.