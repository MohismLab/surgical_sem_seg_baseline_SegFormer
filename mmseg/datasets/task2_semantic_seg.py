from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Task2SemanticSegDataset(CustomDataset):
    """
    Task2 surgical semantic segmentation dataset.
        
        注意格式是：
        segformer_data/
        ├── images/{train,val,test}/<stem>.png
        ├── annotations/{train,val,test}/<stem>.png   # uint8, values 0..29
        └── folds/fold_<k>/{train,val}.txt            # one <stem> per line
    """

    CLASSES = (
        'background',
        'Laparoscopic Needle Driver-Tool Clasper',
        'Laparoscopic Needle Driver-Tool Shaft',
        'Laparoscopic Scissor-Tool Clasper',
        'Laparoscopic Scissor-Tool Shaft',
        'Hem-o-lock Applier-Tool Clasper',
        'Hem-o-lock Applier-Tool Shaft',
        'Bulldog Applier-Tool Clasper',
        'Bulldog Applier-Tool Shaft',
        'Robotic Needle Driver-Left-Tool Clasper',
        'Robotic Needle Driver-Left-Tool Wrist',
        'Robotic Needle Driver-Left-Tool Shaft',
        'Robotic Needle Driver-Right-Tool Clasper',
        'Robotic Needle Driver-Right-Tool Wrist',
        'Robotic Needle Driver-Right-Tool Shaft',
        'Robotic Prograsp Forceps-Tool Clasper',
        'Robotic Prograsp Forceps-Tool Wrist',
        'Robotic Prograsp Forceps-Tool Shaft',
        'Robotic Monopolar curved scissor-Tool Clasper',
        'Robotic Monopolar curved scissor-Tool Shaft',
        'Suction Tool',
        'Suturing Needle',
        'Thread',
        'Bulldog',
        'Hem-o-lock',
        'Trocar',
        'Renal Wound',
        'Kidney',
        'Resected Tumor',
        'class_29',
    )

    PALETTE = [
        [0, 0, 0],
        [51, 255, 255],
        [0, 51, 204],
        [51, 255, 255],
        [0, 51, 204],
        [51, 255, 255],
        [0, 51, 204],
        [51, 255, 255],
        [0, 51, 204],
        [51, 255, 255],
        [0, 255, 0],
        [0, 51, 204],
        [51, 255, 255],
        [0, 255, 0],
        [0, 51, 204],
        [51, 255, 255],
        [0, 255, 0],
        [0, 51, 204],
        [51, 255, 255],
        [0, 51, 204],
        [51, 153, 255],
        [255, 255, 0],
        [255, 204, 0],
        [51, 51, 102],
        [204, 51, 204],
        [255, 204, 204],
        [204, 0, 0],
        [255, 102, 102],
        [85, 0, 127],
        [128, 128, 128],
    ]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
