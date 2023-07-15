import os

def get_dataset_path(name):
    if name == 'coco':
        """
        ── annotations
        │   ├── instances_train2017.json
        │   ├── instances_val2017.json
        ├── test2017
        │   └── *.jpg
        ├── train2017
        │   └── *.jpg
        └── val2017
            └── *.jpg

        """
        return '/data_path/coco_2017'

    if name == 'PCD_raw':
        """
        ├── GSV
        │   ├── mask
        │   │   └── *.png
        │   ├── README.txt
        │   ├── t0
        │   │   └── *.jpg
        │   └── t1
        │       └── *.jpg
        └── TSUNAMI
            ├── mask
            │   └── *.png
            ├── README.txt
            ├── t0
            │   └── *.jpg
            └── t1
                └── *.jpg
        """
        return ''

    if name == 'PCD_CV':
        """
        ├── set0
        │   ├── test
        │   │   ├── mask
        │   │   ├── t0
        │   │   └── t1
        │   ├── test.txt
        │   ├── train
        │   │   ├── mask
        │   │   ├── t0
        │   │   └── t1
        │   └── train.txt
        ├── set1
        ├── set2
        ├── set3
        └── set4
        """
        return '/Users/chikaagbakuru/Documents/C-3po/pcd'

    if name == 'CMU_binary':
        """
        ├── test
        │   ├── mask
        │   ├── t0
        │   └── t1
        └── train
            ├── mask
            ├── t0
            └── t1
        """
        return '//Users/chikaagbakuru/Documents/C-3po/VL-CMU-CD-binary_mask'

    if name == 'CMU_raw':
        """
        ├── raw
        ├── test
        └── train
        """
        return '/data_path/VL-CMU-CD'

    if name == 'ChangeSim':
        """
        ├── Query
        │   ├── Query_Seq_Test
        │   └── Query_Seq_Train
        └── Reference
            ├── Ref_Seq_Test
            └── Ref_Seq_Train
        """
        return '/data_path/ChangeSim'
