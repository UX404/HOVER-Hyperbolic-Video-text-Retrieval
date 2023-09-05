# 🗂 Data files

Data files are arranged in this way:

```bash
|-- anet
    |-- videos (original ActivityNet videos)
    |-- videos_compressed (compressed videos)
    |-- train_tree_deep.json
    |-- train_tree_shallow.json
    |-- train_tree_fewshot.json
    |-- test_tree_deep.json
    |-- test_tree_shallow.json
    |-- val_tree_deep.json
    |-- val_tree_shallow.json
|-- charades
    |-- videos
    |-- videos_compressed
    |-- train_tree_deep.json
    |-- train_tree_shallow.json
    |-- test_tree_deep.json
    |-- test_tree_shallow.json
|-- msrvtt_data
    |-- MSRVTT_Videos
    |-- videos_compressed
    |-- MSRVTT_data.json
    |-- MSRVTT_JSFUSION_test.csv
    |-- MSRVTT_train.7k.csv
    |-- MSRVTT_train.9k.csv
```

About the compressed videos, see [Clip4CLIP](https://github.com/ArrowLuo/CLIP4Clip#compress-video-for-speed-up-optional).
