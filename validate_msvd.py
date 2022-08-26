import torch
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored

from DataLoader import VideoQADataLoader, VideoQADataLoader_oie
from utils import todevice

import model.HCRN as HCRN

from config import cfg, cfg_from_file


def validate(cfg, model, data, device, write_preds=False):
    model.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
        what_acc,who_acc,how_acc,when_acc,where_acc = 0.,0.,0.,0.,0.
        what_count, who_count, how_count, when_count, where_count = 0,0,0,0,0

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            logits = model(*batch_input)
            
            preds = logits.detach().argmax(1)
            agreeings = (preds == answers)
            if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
                what_idx = []
                who_idx = []
                how_idx = []
                when_idx = []
                where_idx = []

                key_word = batch_input[-10][:,0].to('cpu') # batch-based questions word
                for i,word in enumerate(key_word):
                    word = int(word)
                    if data.vocab['question_idx_to_token'][word] == 'what':
                        what_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'who':
                        who_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'how':
                        how_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'when':
                        when_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'where':
                        where_idx.append(i)
            
            if write_preds:
                preds = logits.argmax(1)
                answer_vocab = data.vocab['answer_idx_to_token']

                for predict in preds:
                    all_preds.append(answer_vocab[predict.item()])
                
                for gt in answers:
                    gts.append(answer_vocab[gt.item()])
                
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id)

            if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)

                what_acc += agreeings.float()[what_idx].sum().item() if what_idx != [] else 0
                who_acc += agreeings.float()[who_idx].sum().item() if who_idx != [] else 0
                how_acc += agreeings.float()[how_idx].sum().item() if how_idx != [] else 0
                when_acc += agreeings.float()[when_idx].sum().item() if when_idx != [] else 0
                where_acc += agreeings.float()[where_idx].sum().item() if where_idx != [] else 0
                what_count += len(what_idx)
                who_count += len(who_idx)
                how_count += len(how_idx)
                when_count += len(when_idx)
                where_count += len(where_idx)

        acc = total_acc / count
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            what_acc = what_acc / what_count
            who_acc = who_acc / who_count
            how_acc = how_acc / how_count
            when_acc = when_acc / when_count
            where_acc = where_acc / where_count
   
    if not write_preds:
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            return acc, what_acc, who_acc, how_acc, when_acc, where_acc
    else:
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            return acc, all_preds, gts, v_ids, q_ids, what_acc, who_acc, how_acc, when_acc, where_acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/sutd-qa.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['sutd-qa','tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat_swin_large.h5'
        cfg.dataset.motion_feat = '{}_motion_feat_swin_large.h5'
        cfg.dataset.appearance_dict = '{}_appearance_feat_swin_large_dict.h5'
        cfg.dataset.motion_dict = '{}_motion_feat_swin_large_dict.h5'
        cfg.dataset.vocab_json = '{}_vocabv2.json'
        cfg.dataset.vocab_subject_json = '{}_vocab_subjectv2.json'
        cfg.dataset.vocab_relation_json = '{}_vocab_relationv2.json'
        cfg.dataset.vocab_object_json = '{}_vocab_objectv2.json'
        cfg.dataset.test_question_pt = '{}_test_questionsv2.pt'
        cfg.dataset.test_question_subject_pt = '{}_test_questions_subjectv2.pt'
        cfg.dataset.test_question_relation_pt = '{}_test_questions_relationv2.pt'
        cfg.dataset.test_question_object_pt = '{}_test_questions_objectv2.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.test_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_subject_pt.format(cfg.dataset.name))
        cfg.dataset.test_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_relation_pt.format(cfg.dataset.name))
        cfg.dataset.test_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_object_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))
        cfg.dataset.vocab_subject_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_subject_json.format(cfg.dataset.name))
        cfg.dataset.vocab_relation_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_relation_json.format(cfg.dataset.name))
        cfg.dataset.vocab_object_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_object_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))
        cfg.dataset.appearance_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_dict.format(cfg.dataset.name))
        cfg.dataset.motion_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_dict.format(cfg.dataset.name))


    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'question_subject_pt': cfg.dataset.test_question_subject_pt,
        'question_relation_pt': cfg.dataset.test_question_relation_pt,
        'question_object_pt': cfg.dataset.test_question_object_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'vocab_subject_json': cfg.dataset.vocab_subject_json,
        'vocab_relation_json': cfg.dataset.vocab_relation_json,
        'vocab_object_json': cfg.dataset.vocab_object_json,  
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'appearance_dict': cfg.dataset.appearance_dict,
        'motion_dict': cfg.dataset.motion_dict,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoader_oie(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.STC_Transformer(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids = validate(cfg, model, test_loader, device, cfg.test.write_preds)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")

        if cfg.dataset.question_type in ['action', 'transition']: \
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], ans_candidates[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': dict[str(q_id)][0], 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 samples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 examples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
