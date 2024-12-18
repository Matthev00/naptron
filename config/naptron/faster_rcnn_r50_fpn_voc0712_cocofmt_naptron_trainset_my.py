_base_ = [
    './faster_rcnn_r50_fpn_voc0712_cocofmt_naptron.py'
]

data_root = 'data/nuimages/'

data = dict(
    samples_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/nuimages_v1.0-train_car_filtered_train.json',
        img_prefix=''
    ),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/nuimages_v1.0-train_car_filtered_test.json',
        img_prefix=''
    ),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/nuimages_v1.0-train_car_filtered_val.json',
        img_prefix=''
    )
)