import os
import json
import numpy as np
import torch
import random


def wipe_aux(txt):
    replacements_1 = [' a ', ' the ', ' an ', ' both ', ' is ', ' are ']
    replacements_2 = ['A ', 'The ', 'An ', 'Both', 'Is', 'Are']
    for word in replacements_1:
        while word in txt:
            txt = txt.replace(word, ' ')
    for word in replacements_2:
        while word in txt:
            txt = txt.replace(word, '')
    return txt


def charades_txt2json(data_dir, split='train'):
    anno_file = open(os.path.join(data_dir, "charades_sta_{}.txt".format(split)),'r')
    annotations = {}
    for line in anno_file:
        anno, sent = line.split("##")
        sent = sent.split('.\n')[0]
        vid, s_time_str, e_time_str = anno.split(" ")
        s_time = float(s_time_str)
        e_time = float(e_time_str)
        if s_time >= e_time:
            continue
        if not vid in annotations:
            annotations[vid] = []
        annotations[vid].append([[s_time, e_time], sent+'.'])

    for k, v in annotations.items():
        idxs = torch.argsort(torch.tensor([i[0][0] for i in v]))
        annotations[k] = [v[idx] for idx in idxs]
    for k, v in annotations.items():
        timestamps = [i[0] for i in v]
        sentences = [i[1] for i in v]
        annotations[k] = {
            "duration": -1, 
            "timestamps": timestamps, 
            "sentences":sentences
        }

    new_anno_file = open(os.path.join(data_dir, "{}.json".format(split)), 'w')
    json.dump(annotations, new_anno_file)


def charades_json2txt(data_dir, split='train', filename='tree_deep'):
    with open(os.path.join(data_dir, '{}_{}.json'.format(split, filename)),'r') as f:
        annotations = json.load(f)
    new_anno = ''
    for k, v in annotations.items():
        for ts, txt in zip(v['timestamps'], v['sentences']):
            new_anno += '%s %.1f %.1f##%s\n' % (k, ts[0], ts[1], txt)
    with open(os.path.join(data_dir, '{}_{}.txt'.format(split, filename)),'w') as f:
        f.write(new_anno)

def synthetise_charades(data_dir, split='train'):
    new_anno = ''
    anno_file = open(os.path.join(data_dir, "charades_sta_{}.txt".format(split)),'r')
    cur_vid = None

    for line in anno_file:
        anno, sent = line.split("##")
        sent = sent.split('.\n')[0]
        vid, s_time_str, e_time_str = anno.split(" ")
        s_time = float(s_time_str)
        e_time = float(e_time_str)

        if vid != cur_vid:
            cur_vid = vid
            vid_anno = {}
            tree_found = False
        if tree_found:
            continue
        if s_time < e_time:
            if not vid in vid_anno:
                vid_anno[vid] = []
            for moment in vid_anno[vid]:
                if abs(s_time - moment['times'][0]) < 1 or abs(e_time - moment['times'][1]) < 1:
                    continue
                if moment['times'][0] <= s_time <= moment['times'][1]+2 and e_time > moment['times'][1]:
                    new_anno += '%s %s %s##%s, then %s.\n' % (vid, moment['s_time'], e_time, moment['text'], sent)
                    new_anno += moment['line']
                    new_anno += line
                    tree_found = True
                    break
                if moment['times'][0]-2 <= e_time <= moment['times'][1] and s_time < moment['times'][0]:
                    new_anno += '%s %s %s##%s, then %s.\n' % (vid, s_time, moment['e_time'], sent, moment['text'])
                    new_anno += line
                    new_anno += moment['line']
                    tree_found = True
                    break
            vid_anno[vid].append({
                'video': vid, 
                'text': sent,
                'times': [s_time, e_time],
                's_time': s_time_str,
                'e_time': e_time_str,
                'line': line,
            })
    anno_file.close()

    new_anno_file = open(os.path.join(data_dir, "charades_tree_{}.txt".format(split)),'w')
    new_anno_file.write(new_anno[:-1])
    new_anno_file.close()


def synthetise_anet_simple(data_dir, split='train'):
    with open(os.path.join(data_dir, '{}.json'.format(split)),'r') as f:
        annotations = json.load(f)
    anno_pairs = []
    cnt = 0
    for vid, video_anno in annotations.items():
        duration = video_anno['duration']
        # for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
        moment_duration = np.array([t[1]-t[0] for t in video_anno['timestamps']])
        cnt += len(video_anno['sentences'])
        idx = np.argmax(moment_duration)
        timestamp = video_anno['timestamps'][idx]
        sentence = video_anno['sentences'][idx]
        anno_pairs.append(
            {
                'video': vid,
                'text': sentence,
                'duration': duration,
                'times': [max(timestamp[0],0)/duration ,min(timestamp[1],duration)/duration],
                'original_times': [timestamp[0], timestamp[1]]
            }
        )

    new_anno = {}
    for anno in anno_pairs:
        new_anno[anno['video']] = {
            'duration': anno['duration'],
            'timestamps': [anno['original_times']],
            'sentences': [anno['text']]
        }
    new_anno_file = open(os.path.join(data_dir, "{}_single.json".format(split)), 'w')
    json.dump(new_anno, new_anno_file)
    print('%s: %d -> %d' % (split, cnt, len(new_anno)))


def synthetise_anet_single(data_dir, split='train'):
    with open(os.path.join(data_dir, '{}.json'.format(split)),'r') as f:
        annotations = json.load(f)
    new_anno = {}
    scnt = 0
    for vid, video_anno in annotations.items():
        synthesis_list = []
        tss = video_anno['timestamps']
        txts = video_anno['sentences']

        # for n, (timestamp, sentence) in enumerate(zip(video_anno['timestamps'], video_anno['sentences'])):
        #     if n < len(video_anno['sentences'])-1 and \
        #         timestamp[1]-2 < video_anno['timestamps'][n+1][0] < timestamp[1]+2 and \
        #             timestamp[0] < video_anno['timestamps'][n+1][0]:
        #             synthesis_list.append((video_anno['timestamps'][n+1][1] - timestamp[0], n))

        # single
        for n in range(len(tss)):
            synthesis_single = [tss[n][0], tss[n][0], tss[n][1], [n], tss[n][1]-tss[n][0]]  # [start_ts, last_ts, end_te, idxs, duration]
            for m in range(n+1, len(tss)):
                # print(synthesis_single, tss[m])
                if synthesis_single[2] < tss[m][0]:
                    synthesis_single[1] = tss[m][0]
                    synthesis_single[2] = tss[m][1]
                    synthesis_single[3].append(m)
                    synthesis_single[4] += (tss[m][1] - tss[m][0])
            if len(synthesis_single[3]) > 1:
                synthesis_list.append(synthesis_single)
        if len(synthesis_list) == 0:
            continue

        duration = video_anno['duration']
        moment_duration = np.array([t[4] for t in synthesis_list])
        idx = np.argmax(moment_duration)
        idxs = synthesis_list[idx][3]
        scnt += 1

        syn_ts = [synthesis_list[idx][0], synthesis_list[idx][2]]
        syn_txt = txts[idxs[0]]
        for idx in idxs[1:]:
            # syn_txt += ' Then '
            syn_txt += txts[idx]

        for word in replacements_1:
            while word in syn_txt:
                syn_txt = syn_txt.replace(word, ' ')
        for word in replacements_2:
            while word in syn_txt:
                syn_txt = syn_txt.replace(word, '')
        
        child_duration = np.array([tss[idx][1] - tss[idx][0] for idx in idxs])
        child_idx = np.argmax(child_duration)
        new_anno[vid] = {
            'duration': duration,
            'timestamps': [syn_ts] + [tss[idxs[child_idx]]],
            'sentences': [syn_txt] + [txts[idxs[child_idx]]],
        }
            
        
    new_anno_file = open(os.path.join(data_dir, "{}_tree_s.json".format(split)), 'w')
    json.dump(new_anno, new_anno_file)
    print('%s: s %d' % (split, scnt))


def synthetise_anet_multiple(data_dir, split='train'):
    with open(os.path.join(data_dir, '{}.json'.format(split)),'r') as f:
        annotations = json.load(f)
    new_anno = {}
    cnt = 0
    for vid, video_anno in annotations.items():
        synthesis_list = []
        tss = video_anno['timestamps']
        txts = video_anno['sentences']

        # for n, (timestamp, sentence) in enumerate(zip(video_anno['timestamps'], video_anno['sentences'])):
        #     if n < len(video_anno['sentences'])-1 and \
        #         timestamp[1]-2 < video_anno['timestamps'][n+1][0] < timestamp[1]+2 and \
        #             timestamp[0] < video_anno['timestamps'][n+1][0]:
        #             synthesis_list.append((video_anno['timestamps'][n+1][1] - timestamp[0], n))

        for n in range(len(tss)):
            synthesis_single = [tss[n][0], tss[n][0], tss[n][1], [n]]  # [start_ts, last_ts, end_te, idxs]
            for m in range(n+1, len(tss)):
                # print(synthesis_single, tss[m])
                if synthesis_single[2]-3 < tss[m][0] < synthesis_single[2]+3 and \
                synthesis_single[1] < tss[m][0]:
                    synthesis_single[1] = tss[m][0]
                    synthesis_single[2] = tss[m][1]
                    synthesis_single[3].append(m)
            if len(synthesis_single[3]) > 1:
                synthesis_list.append(synthesis_single)
        if len(synthesis_list) == 0:
            continue

        duration = video_anno['duration']
        moment_duration = np.array([t[2] - t[0] for t in synthesis_list])
        idx = np.argmax(moment_duration)
        idxs = synthesis_list[idx][3]
        cnt += len(idxs)

        syn_ts = [synthesis_list[idx][0], synthesis_list[idx][2]]
        syn_txt = txts[idxs[0]]
        for idx in idxs[1:]:
            # syn_txt += ' Then '
            syn_txt += txts[idx]

        for word in replacements_1:
            while word in syn_txt:
                syn_txt = syn_txt.replace(word, ' ')
        for word in replacements_2:
            while word in syn_txt:
                syn_txt = syn_txt.replace(word, '')

        new_anno[vid] = {
            'duration': duration,
            'timestamps': [syn_ts] + [tss[idx] for idx in idxs],
            'sentences': [syn_txt] + [txts[idx] for idx in idxs],
        }
    new_anno_file = open(os.path.join(data_dir, "{}_tree_m.json".format(split)), 'w')
    json.dump(new_anno, new_anno_file)
    print('%s: m %d/%d' % (split, cnt, len(new_anno)))


def synthetise_anet_shallow(data_dir, split='train'):
    with open(os.path.join(data_dir, '{}.json'.format(split)),'r') as f:
        annotations = json.load(f)
    new_anno = {}
    mcnt = 0
    scnt = 0
    for vid, video_anno in annotations.items():
        synthesis_list = []
        tss = video_anno['timestamps']
        txts = video_anno['sentences']

        # for n, (timestamp, sentence) in enumerate(zip(video_anno['timestamps'], video_anno['sentences'])):
        #     if n < len(video_anno['sentences'])-1 and \
        #         timestamp[1]-2 < video_anno['timestamps'][n+1][0] < timestamp[1]+2 and \
        #             timestamp[0] < video_anno['timestamps'][n+1][0]:
        #             synthesis_list.append((video_anno['timestamps'][n+1][1] - timestamp[0], n))

        # multiple
        for n in range(len(tss)):
            synthesis_single = [tss[n][0], tss[n][0], tss[n][1], [n]]  # [start_ts, last_ts, end_te, idxs]
            for m in range(n+1, len(tss)):
                # print(synthesis_single, tss[m])
                if synthesis_single[2]-3 < tss[m][0] < synthesis_single[2]+3 and \
                synthesis_single[1] < tss[m][0]:
                    synthesis_single[1] = tss[m][0]
                    synthesis_single[2] = tss[m][1]
                    synthesis_single[3].append(m)
            if len(synthesis_single[3]) > 1:
                synthesis_list.append(synthesis_single)
        if len(synthesis_list) != 0:
            duration = video_anno['duration']
            moment_duration = np.array([t[2] - t[0] for t in synthesis_list])
            idx = np.argmax(moment_duration)
            idxs = synthesis_list[idx][3]
            mcnt += len(idxs)

            syn_ts = [synthesis_list[idx][0], synthesis_list[idx][2]]
            syn_txt = txts[idxs[0]]
            for idx in idxs[1:]:
                # syn_txt += ' '
                syn_txt += txts[idx]

            syn_txt = wipe_aux(syn_txt)
            
            new_anno[vid] = {
                'duration': duration,
                'timestamps': [syn_ts] + [tss[idx] for idx in idxs],
                'sentences': [syn_txt] + [txts[idx] for idx in idxs],
            }

        # single
        else:
            for n in range(len(tss)):
                synthesis_single = [tss[n][0], tss[n][0], tss[n][1], [n], tss[n][1]-tss[n][0]]  # [start_ts, last_ts, end_te, idxs, duration]
                for m in range(n+1, len(tss)):
                    # print(synthesis_single, tss[m])
                    if synthesis_single[2] < tss[m][0]:
                        synthesis_single[1] = tss[m][0]
                        synthesis_single[2] = tss[m][1]
                        synthesis_single[3].append(m)
                        synthesis_single[4] += (tss[m][1] - tss[m][0])
                if len(synthesis_single[3]) > 1:
                    synthesis_list.append(synthesis_single)
            if len(synthesis_list) == 0:
                continue

            duration = video_anno['duration']
            moment_duration = np.array([t[4] for t in synthesis_list])
            idx = np.argmax(moment_duration)
            idxs = synthesis_list[idx][3]
            scnt += 1

            syn_ts = [synthesis_list[idx][0], synthesis_list[idx][2]]
            syn_txt = txts[idxs[0]]
            for idx in idxs[1:]:
                # syn_txt += ' '
                syn_txt += txts[idx]

            syn_txt = wipe_aux(syn_txt)
            
            child_duration = np.array([tss[idx][1] - tss[idx][0] for idx in idxs])
            child_idx = np.argmax(child_duration)
            new_anno[vid] = {
                'duration': duration,
                'timestamps': [syn_ts] + [tss[idxs[child_idx]]],
                'sentences': [syn_txt] + [txts[idxs[child_idx]]],
            }
            
    # for k in np.random.choice(list(new_anno), int(len(new_anno) * (1-1/10)), replace=False):
    #     del new_anno[k]
    # print(len(new_anno))
    new_anno_file = open(os.path.join(data_dir, "{}_tree_shallow.json".format(split)), 'w')
    json.dump(new_anno, new_anno_file)
    print('%s: m %d/%d, s %d' % (split, mcnt, len(new_anno)-scnt, scnt))


def synthetise_anet_deep(data_dir, split='train'):
    with open(os.path.join(data_dir, '{}.json'.format(split)),'r') as f:
        annotations = json.load(f)
    new_anno = {}
    mcnt = 0
    scnt = 0
    height_cnt = [0] * 8
    for vid, video_anno in annotations.items():
        synthesis_list = []
        tss = video_anno['timestamps']
        txts = video_anno['sentences']
        # if random.random() > 0.1:
        #     continue

        # multiple
        for n in range(len(tss)):
            synthesis_single = [tss[n][0], tss[n][0], tss[n][1], [n]]  # [start_ts, last_ts, end_te, idxs]
            for m in range(n+1, len(tss)):
                # print(synthesis_single, tss[m])
                if synthesis_single[2]-3 < tss[m][0] < synthesis_single[2]+3 and \
                synthesis_single[1] < tss[m][0]:
                    synthesis_single[1] = tss[m][0]
                    synthesis_single[2] = tss[m][1]
                    synthesis_single[3].append(m)
            if len(synthesis_single[3]) > 1:
                synthesis_list.append(synthesis_single)
        if len(synthesis_list) != 0:
            duration = video_anno['duration']
            moment_duration = np.array([t[2] - t[0] for t in synthesis_list])
            idx = np.argmax(moment_duration)
            leaf_idxs = synthesis_list[idx][3]
            leaf_tss = [tss[idx] for idx in leaf_idxs]
            leaf_txts = [txts[idx] for idx in leaf_idxs]

            subtree_cnt = 0
            height = 1
            while len(leaf_tss) != 1:
                leaf_tss = [leaf_tss[min(n, len(leaf_tss)-2): min(n, len(leaf_tss)-2)+2] for n in range(0, len(leaf_tss), 2)]  # group into 2/3 children
                # if len(leaf_tss[-1]) == 1:
                #     leaf_tss[-2] += leaf_tss[-1]
                #     leaf_tss = leaf_tss[:-1]
                leaf_txts = [leaf_txts[min(n, len(leaf_txts)-2): min(n, len(leaf_txts)-2)+2] for n in range(0, len(leaf_txts), 2)]
                # if len(leaf_txts[-1]) == 1:
                #     leaf_txts[-2] += leaf_txts[-1]
                #     leaf_txts = leaf_txts[:-1]
                parent_leaf_tss = []
                parent_leaf_txts = []
                for child_leaf_tss, child_leaf_txts in zip(leaf_tss, leaf_txts):
                    parent_leaf_ts = [child_leaf_tss[0][0], child_leaf_tss[-1][1]]
                    # parent_leaf_txt = ' '.join(child_leaf_txts)
                    parent_leaf_txt = wipe_aux(''.join(child_leaf_txts))
                    new_anno['%s#%d' % (vid, subtree_cnt)] = {
                        'duration': duration,
                        'timestamps': [parent_leaf_ts] + child_leaf_tss,
                        'sentences': [parent_leaf_txt] + child_leaf_txts,
                    }
                    parent_leaf_tss.append(parent_leaf_ts)
                    parent_leaf_txts.append(parent_leaf_txt)
                    mcnt += len(child_leaf_tss)
                    subtree_cnt += 1
                # print(leaf_tss)
                # print(leaf_txts)
                # input()
                leaf_tss = parent_leaf_tss
                leaf_txts = parent_leaf_txts
                height += 1
            mcnt += 1
            height_cnt[height-1] += 1
            
            

        # single
        else:
            for n in range(len(tss)):
                synthesis_single = [tss[n][0], tss[n][0], tss[n][1], [n], tss[n][1]-tss[n][0]]  # [start_ts, last_ts, end_te, idxs, duration]
                for m in range(n+1, len(tss)):
                    # print(synthesis_single, tss[m])
                    if synthesis_single[2] < tss[m][0]:
                        synthesis_single[1] = tss[m][0]
                        synthesis_single[2] = tss[m][1]
                        synthesis_single[3].append(m)
                        synthesis_single[4] += (tss[m][1] - tss[m][0])
                if len(synthesis_single[3]) > 1:
                    synthesis_list.append(synthesis_single)
            if len(synthesis_list) == 0:
                continue

            duration = video_anno['duration']
            moment_duration = np.array([t[4] for t in synthesis_list])
            idx = np.argmax(moment_duration)
            idxs = synthesis_list[idx][3]
            scnt += 1

            syn_ts = [synthesis_list[idx][0], synthesis_list[idx][2]]
            syn_txt = txts[idxs[0]]
            for idx in idxs[1:]:
                # syn_txt += ' '
                syn_txt += txts[idx]

            syn_txt = wipe_aux(syn_txt)
            
            child_duration = np.array([tss[idx][1] - tss[idx][0] for idx in idxs])
            child_idx = np.argmax(child_duration)
            new_anno[vid] = {
                'duration': duration,
                'timestamps': [syn_ts] + [tss[idxs[child_idx]]],
                'sentences': [syn_txt] + [txts[idxs[child_idx]]],
            }
            height_cnt[1] += 1
            
    # for k in np.random.choice(list(new_anno), int(len(new_anno) * (1-1/10)), replace=False):
    #     del new_anno[k]
    # print(len(new_anno))
    new_anno_file = open(os.path.join(data_dir, "{}_tree_deep.json".format(split)), 'w')
    # new_anno_file = open(os.path.join(data_dir, "{}_tree_deepfewshot.json".format(split)), 'w')
    json.dump(new_anno, new_anno_file)
    print('%s: m nodes:%d/subtrees:%d/trees:%d, s %d' % (split, mcnt, len(new_anno)-scnt, sum(height_cnt)-scnt, scnt))
    print('height_cnt:', height_cnt)


if __name__ == '__main__':
    # file = '/storage_fast/rqshi/Anet'
    file = '/storage_fast/rqshi/Charades-STA'
    # splits = ['test', 'val', 'train']
    splits = ['test', 'train']
    for split in splits:
        # charades_txt2json(file, split)
        # synthetise_anet_deep(file, split)
        charades_json2txt(file, split, filename='tree_deepnodes')