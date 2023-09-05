from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor

# This is to sample a small amount of trees for pre-experiments. 0 means do not sample.
sample = 0
child_num = 0
CHILDLIMIT = 6

class Anet_DataLoader(Dataset):
    """Anet dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_dir = '/storage_fast/rqshi/Anet'
        self.split = csv_path
        self.num_chunks = 16

        self.data = self._load_metadata()
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _load_metadata(self):
        split = self.split.replace('val', 'test')
        with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            if child_num != 0 and child_num != CHILDLIMIT and len(video_anno['sentences']) != child_num + 1:
                continue
            if child_num == CHILDLIMIT and len(video_anno['sentences']) <= CHILDLIMIT:
                continue
            single_anno = []
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                single_anno.append({
                    'video': vid.split('#')[0],
                    'text': sentence,
                    'duration': duration,
                    'times': [max(timestamp[0],0)/duration ,min(timestamp[1],duration)/duration],
                    'original_times': [max(timestamp[0],0) ,min(timestamp[1],duration)]
                })
            anno_pairs.append(single_anno)
        
        if sample:
            if sample < 1:
                idx = np.random.choice(len(anno_pairs), int(sample*len(anno_pairs)))
            else:
                idx = np.random.choice(len(anno_pairs), sample)
            anno_pairs = [anno_pairs[i] for i in idx]

        annotations = []
        for anno in anno_pairs:
            for n in range(1, len(anno)):
                annotations.append([anno[0], anno[n]])
        # print(list(set([anno[0]['video'][-3:] for anno in annotations])))

        # for t in annotations[:7]:
        #     print(t)
        #     print()
        # exit()
        return annotations

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids, times):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            # print(video_path)
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                # print(raw_video_slice.shape)  torch.Size([19, 1, 3, 224, 224])
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    elif 'test' in self.split or 'val' in self.split or 'demo' in self.split:
                        sample_indx = np.linspace(int(raw_video_slice.shape[0] * times[0]), min(int(raw_video_slice.shape[0] * times[1]), raw_video_slice.shape[0]-1), num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                    elif 'train' in self.split:
                        sample_indx = np.linspace(int(raw_video_slice.shape[0] * times[0]), min(int(raw_video_slice.shape[0] * times[1]), raw_video_slice.shape[0]-1), num=self.max_frames + 1, dtype=int)
                        try:
                            ranges = []
                            for idx, interv in enumerate(sample_indx[:-1]):
                                ranges.append((interv, sample_indx[idx + 1]))
                            frame_idxs = [random.choice(range(x[0], x[1]+1)) for x in ranges]
                        except:
                            print(video_id, times, sample_indx)
                            frame_idxs = np.linspace(int(raw_video_slice.shape[0] * times[0]), min(int(raw_video_slice.shape[0] * times[1]), raw_video_slice.shape[0]-1), num=self.max_frames, dtype=int)
                        # print(times, frame_idxs)
                        video_slice = raw_video_slice[frame_idxs, ...]
                    else:
                        print('not this split:', self.split)
                        exit(-1)
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                # print(video_slice.shape)  torch.Size([12, 1, 3, 224, 224])
                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
                # print('end', video.shape)     end (1, 12, 1, 3, 224, 224)
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        # print(self.data[idx][0]['video'])
        pairs_texts, pairs_masks, pairs_segments, videos, video_masks = [], [], [], [], []
        # print(idx, self.data[idx][0]['video'], self.data[idx][1]['video'])
        for clip in self.data[idx]:
            video_id = clip['video']
            sentence = clip['text']
            times = clip['times']
            # print(video_id, sentence, times)
            # 9JZO2 person holding a cup in the doorway
            # video4888 a monkey is fed on a towel on top of a dryer

            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
            video, video_mask = self._get_rawvideo(choice_video_ids, times)
            # print(pairs_text.shape, pairs_mask.shape, pairs_segment.shape, video.shape, video_mask.shape)
            # (1, 32) (1, 32) (1, 32) (1, 12, 1, 3, 224, 224) (1, 7)
            pairs_texts.append(pairs_text)
            pairs_masks.append(pairs_mask)
            pairs_segments.append(pairs_segment)
            videos.append(video)
            video_masks.append(video_mask)
        
        pairs_texts = np.stack(pairs_texts)
        pairs_masks = np.stack(pairs_masks)
        pairs_segments = np.stack(pairs_segments)
        videos = np.stack(videos)
        video_masks = np.stack(video_masks)
        # print(pairs_texts.shape, pairs_masks.shape, pairs_segments.shape, videos.shape, video_masks.shape)
        # (3, 1, 32) (3, 1, 32) (3, 1, 32) (3, 1, 4, 1, 3, 224, 224) (3, 1, 7)
        # (2, 1, 72) (2, 1, 72) (2, 1, 72) (2, 1, 12, 1, 3, 224, 224) (2, 1, 12)
        return pairs_texts, pairs_masks, pairs_segments, videos, video_masks


class Anet_Test_DataLoader(Dataset):
    """Anet test dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_dir = '/storage_fast/rqshi/Anet'
        self.split = csv_path
        self.num_chunks = 16

        self.data = self._load_metadata()
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _load_metadata(self):
        split = self.split.replace('val', 'test')
        with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        if sample:
            if sample < 1:
                keys = np.random.choice(list(annotations.keys()), int(sample*len(annotations)))
            else:
                keys = np.random.choice(list(annotations.keys()), sample)
            annotations = {k: annotations[k] for k in keys}

        anno_pairs = []
        node_cnt = []
        for vid, video_anno in annotations.items():
            if child_num != 0 and child_num != CHILDLIMIT and len(video_anno['sentences']) != child_num + 1:
                continue
            if child_num == CHILDLIMIT and len(video_anno['sentences']) <= CHILDLIMIT:
                continue
            node_cnt.append(len(video_anno['sentences']))
            duration = video_anno['duration']
            for n, (timestamp, sentence) in enumerate(zip(video_anno['timestamps'], video_anno['sentences'])):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid.split('#')[0],
                            'node': n,
                            'text': sentence,
                            'duration': duration,
                            'times': [max(timestamp[0],0)/duration ,min(timestamp[1],duration)/duration],
                            'original_times': [max(timestamp[0],0) ,min(timestamp[1],duration)]
                        }
                    )
        from collections import Counter
        print(Counter(node_cnt))
        # anno_pairs = [anno for anno in anno_pairs if anno['video'][-3:] in ['Z-o', 'djE', 'tt0', 'feg', 'Gdk', 'uwI', '7jc', 'is0', 'oG8', 'vY0', '_Wk', 'etM', 'hes', 'Zgs', '69U', 'xpw']]
        # anno_pairs = [anno for anno in anno_pairs if anno['video'][-3:] in ['TdQ', 'CCU', 'sYA', 'l-Q', 'Z-o', '99E', 'S9o', 'v-w', 'tbg', 'egk', '6iE', 'iCg', 'MGc', 'OvE', '_Qw', 'dtI', 'c9o', 'vRA', 'EB0', '8WQ', 'Rek', 'KyM', 'i28', '_6Q']]
        return anno_pairs

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids, times):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                # print(raw_video_slice.shape)  torch.Size([19, 1, 3, 224, 224])
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    elif 'test' in self.split or 'val' in self.split or 'demo' in self.split:
                        sample_indx = np.linspace(int(raw_video_slice.shape[0] * times[0]), min(int(raw_video_slice.shape[0] * times[1]), raw_video_slice.shape[0]-1), num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                    elif 'train' in self.split:
                        sample_indx = np.linspace(int(raw_video_slice.shape[0] * times[0]), min(int(raw_video_slice.shape[0] * times[1]), raw_video_slice.shape[0]-1), num=self.max_frames + 1, dtype=int)
                        ranges = []
                        for idx, interv in enumerate(sample_indx[:-1]):
                            ranges.append((interv, sample_indx[idx + 1]))
                        frame_idxs = [random.choice(range(x[0], x[1]+1)) for x in ranges]
                        # print(times, frame_idxs)
                        video_slice = raw_video_slice[frame_idxs, ...]
                    else:
                        print('not this split:', self.split)
                        exit(-1)
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                # print(video_slice.shape)  torch.Size([12, 1, 3, 224, 224])
                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
                # print('end', video.shape)     end (1, 12, 1, 3, 224, 224)
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data[idx]['video']
        node = self.data[idx]['node']
        sentence = self.data[idx]['text']
        times = self.data[idx]['times']
        # print(video_id, sentence)
        # 9JZO2 person holding a cup in the doorway
        # video4888 a monkey is fed on a towel on top of a dryer

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids, times)
        # print(pairs_text.shape, pairs_mask.shape, pairs_segment.shape, video.shape, video_mask.shape)
        # (1, 32) (1, 32) (1, 32) (1, 12, 1, 3, 224, 224) (1, 7)
        if self.split == 'demo':
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, (video_id[-3:], node)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask