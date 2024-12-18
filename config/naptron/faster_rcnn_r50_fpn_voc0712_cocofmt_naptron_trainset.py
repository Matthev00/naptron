_base_ = [
    './faster_rcnn_r50_fpn_voc0712_cocofmt_naptron.py'
]

data_root = 'data/nuimages/'

data = dict(
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/nuimages_v1.0-train_car_filtered.json',
        img_prefix=''
    )
)