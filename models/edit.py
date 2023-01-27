"""
schema editing modules
"""
import copy
import torch as th
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

flair.device = th.device('cuda:0')
tagger = SequenceTagger.load("flair/pos-english")
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1', device = "cuda:0")
unmasker = pipeline("fill-mask", device = 0)

def edit_schema_howto(goal, testgoal2traingoals, meta_data, schema_resource, refer_k, replacement, deletion, mask):
    list_schema_steps = []
    list_hops = []
    for k in range(refer_k):
        source, hops = testgoal2traingoals[goal][k]
        list_hops.append(hops)

        target_noun = meta_data[source]['noun']
        current_noun = meta_data[goal]['noun']

        source_steps = schema_resource[source]
        schema_steps = copy.deepcopy(source_steps)

        if replacement == True:
            new_steps = []
            for step in source_steps:
                new_steps.append(step.replace(target_noun.lower(), current_noun.lower()))
            schema_steps = new_steps

        if deletion == True:
            query_embedding_1 = model.encode(source)
            passage_embedding_1 = model.encode(source_steps)
            score_1 = util.dot_score(query_embedding_1, passage_embedding_1)

            query_embedding_2 = model.encode(goal)
            passage_embedding_2 = model.encode(schema_steps)
            score_2 = util.dot_score(query_embedding_2, passage_embedding_2)

            beta = 0.2
            remove_steps = []
            remove_ids = []
            for i, diff in enumerate((score_1 - score_2)[0]):
                if diff / score_1[0][i] > beta:
                    remove_steps.append(schema_steps[i])
                    remove_ids.append(i)

            schema_steps = [step for step in schema_steps if step not in remove_steps]

        if mask == True:
            masked_steps = []
            if len(schema_steps) != 0:
                for step in schema_steps:
                    step_sentence = Sentence(step)
                    tagger.predict(step_sentence)
                    token2tag = {result['text']: result['labels'][0].value for result in step_sentence.to_dict(tag_type='pos')['entities']}
                    token2start_id = {result['text']: result['start_pos'] for result in step_sentence.to_dict(tag_type='pos')['entities']}
                    Ns = []
                    for token, tag in token2tag.items():
                        if tag[:2] == 'NN' and len(token) > 1 and token != current_noun.lower():
                            Ns.append(token)
                            
                    pre_sentences = ["How to " + goal + "? " + step[:token2start_id[token]] for token in Ns]
                    query_embedding = model.encode(pre_sentences)
                    passage_embedding = model.encode(Ns)
                    score = util.dot_score(query_embedding, passage_embedding)
                    token2score = sorted([(Ns[i], float(score[i, i])) for i in range(len(Ns))], key = lambda x: x[1])
                                        
                    current_step = copy.deepcopy(step)
                    for token, score in token2score:
                        current_step_copy = copy.deepcopy(current_step)
                        current_step = re.sub(r'\b{}\b'.format(token), '<mask>', current_step, 1)
                        if current_step == current_step_copy:
                            continue
                        prefix = "How to {}? ".format(goal)
                        mask_result = unmasker(prefix + current_step)[0]
                        if mask_result['token_str'] in current_step_copy:
                            current_step = current_step_copy
                        else:
                            current_step = mask_result['sequence'][len(prefix):]
                    masked_steps.append(current_step)
                schema_steps = copy.deepcopy(masked_steps)
                
        list_schema_steps.append(schema_steps)
    return list_schema_steps, list_hops


def edit_schema(goal, coin_goal2noun, coin_goal2howto_goal, k, howto_goal2steps_induction, howto_goal2verb_noun, replacement, deletion, mask):
    current_noun = coin_goal2noun[goal]
    source, hops = coin_goal2howto_goal[goal][k]
    source_steps = howto_goal2steps_induction[source]
    schema_steps = copy.deepcopy(source_steps)

    try: target_noun = howto_goal2verb_noun[source]['noun'][0]
    except: target_noun = "&&"

    if replacement == True:
        new_steps = []
        for step in schema_steps:
            new_steps.append(step.replace(target_noun.lower(), current_noun.lower()))
        schema_steps = new_steps

    if deletion == True:
        query_embedding_1 = model.encode(source)
        passage_embedding_1 = model.encode(source_steps)
        score_1 = util.dot_score(query_embedding_1, passage_embedding_1)

        query_embedding_2 = model.encode(goal)
        passage_embedding_2 = model.encode(schema_steps)
        score_2 = util.dot_score(query_embedding_2, passage_embedding_2)

        beta = 0.2
        remove_steps = []
        remove_ids = []
        for i, diff in enumerate((score_1 - score_2)[0]):
            if diff / score_1[0][i] > beta:
                remove_steps.append(schema_steps[i])
                remove_ids.append(i)

        schema_steps = [step for step in schema_steps if step not in remove_steps]
        
    if mask == True:
        masked_steps = []
        if len(schema_steps) != 0:
            for step in schema_steps:
                step_sentence = Sentence(step)
                tagger.predict(step_sentence)
                token2tag = {result['text']: result['labels'][0].value for result in step_sentence.to_dict(tag_type='pos')['entities']}
                token2start_id = {result['text']: result['start_pos'] for result in step_sentence.to_dict(tag_type='pos')['entities']}
                Ns = []
                for token, tag in token2tag.items():
                    if tag[:2] == 'NN' and len(token) > 2 and token != current_noun.lower():
                        Ns.append(token)

                pre_sentences = ["How to " + goal + "? " + step[:token2start_id[token]] for token in Ns]
                query_embedding = model.encode(pre_sentences)
                passage_embedding = model.encode(Ns)
                score = util.dot_score(query_embedding, passage_embedding)
                token2score = sorted([(Ns[i], float(score[i, i])) for i in range(len(Ns))], key = lambda x: x[1])

                current_step = copy.deepcopy(step)
                for token, score in token2score:
                    current_step_copy = copy.deepcopy(current_step)
                    current_step = re.sub(r'\b{}\b'.format(token), '<mask>', current_step, 1)
                    if current_step == current_step_copy:
                        continue
                    prefix = "How to {}? ".format(goal)
                    mask_result = unmasker(prefix + current_step)[0]
                    if mask_result['token_str'] in current_step_copy:
                        current_step = current_step_copy
                    else:
                        current_step = mask_result['sequence'][len(prefix):]
                masked_steps.append(current_step)
            schema_steps = copy.deepcopy(masked_steps)

    return schema_steps, hops