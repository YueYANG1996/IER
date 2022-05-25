# IER
Code for the paper: [Induce, Edit, Retrieve: Language Grounded Multimodal Schema for Instructional Video Retrieval](https://arxiv.org/pdf/2111.09276.pdf)

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