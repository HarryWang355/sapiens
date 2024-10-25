custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmseg.engine.optimizers.layer_decay_optim_wrapper',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='GarmentDataPreProcessor')
dataset_train = dict(
    data_root='/data1/datasets/garment-data/iter3-ele0/sapiens-depth-4views',
    serialize_data=False,
    type='DepthGarmentDataset')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=10, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        interval=100,
        max_samples=4,
        type='GarmentDepthVisualizationHook',
        vis_image_height=512,
        vis_image_width=512))
default_scope = 'mmseg'
embed_dim = 1024
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
evaluate_every_n_epochs = 10
image_size = (
    512,
    512,
)
launcher = 'pytorch'
load_from = '/data1/users/yuanhao/sapiens/sapiens_host/sapiens-depth-0.3b/sapiens_0.3b_render_people_epoch_100.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    backbone=dict(
        arch='sapiens_0.3b',
        drop_path_rate=0.0,
        final_norm=True,
        img_size=(
            512,
            512,
        ),
        init_cfg=dict(
            checkpoint=
            '/data1/users/yuanhao/sapiens/sapiens_host/sapiens-pretrain-0.3b/sapiens_0.3b_epoch_1600_clean.pth',
            type='Pretrained'),
        out_type='featmap',
        patch_size=16,
        qkv_bias=True,
        type='mmpretrain.VisionTransformer',
        with_cls_token=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='GarmentDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=384,
        conv_kernel_sizes=(
            1,
            1,
        ),
        conv_out_channels=(
            384,
            384,
        ),
        deconv_kernel_sizes=(
            4,
            4,
            4,
            4,
        ),
        deconv_out_channels=(
            384,
            384,
            384,
            384,
        ),
        in_channels=1024,
        loss_decode=dict(loss_weight=1.0, type='MetricDepthL1Loss'),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=1,
        type='VitDepthHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='DepthEstimator')
model_name = 'sapiens_0.3b'
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_epochs = 100
num_layers = 24
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    constructor='LayerDecayOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0005, type='AdamW', weight_decay=0.1),
    paramwise_cfg=dict(
        custom_keys=dict(
            bias=dict(decay_multi=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)),
        layer_decay_rate=0.85,
        num_layers=24))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=400, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0, by_epoch=True, end=100, eta_min=0.0, power=1.0,
        type='PolyLR'),
]
patch_size = 16
pretrained_checkpoint = '/data1/users/yuanhao/sapiens/sapiens_host/sapiens-pretrain-0.3b/sapiens_0.3b_epoch_1600_clean.pth'
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(type='LoadImage'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackDepthInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        datasets=[
            dict(
                data_root=
                '/data1/datasets/garment-data/iter3-ele0/sapiens-depth-4views',
                serialize_data=False,
                type='DepthGarmentDataset'),
        ],
        metainfo=dict(from_file='configs/_base_/datasets/render_people.py'),
        pipeline=[
            dict(type='LoadImage'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    1.0,
                    2.0,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(type='RandomDepthResizeCompensate'),
            dict(crop_size=(
                512,
                512,
            ), type='RandomDepthCrop'),
            dict(scale=(
                512,
                512,
            ), type='DepthResize'),
            dict(prob=0.5, type='DepthRandomFlip'),
            dict(type='GenerateConditionalDepthTarget'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackConditonalDepthInputs'),
        ],
        type='DepthCombinedDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_datasets = [
    dict(
        data_root=
        '/data1/datasets/garment-data/iter3-ele0/sapiens-depth-4views',
        serialize_data=False,
        type='DepthGarmentDataset'),
]
train_pipeline = [
    dict(type='LoadImage'),
    dict(
        keep_ratio=True,
        ratio_range=(
            1.0,
            2.0,
        ),
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(type='RandomDepthResizeCompensate'),
    dict(crop_size=(
        512,
        512,
    ), type='RandomDepthCrop'),
    dict(scale=(
        512,
        512,
    ), type='DepthResize'),
    dict(prob=0.5, type='DepthRandomFlip'),
    dict(type='GenerateConditionalDepthTarget'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackConditonalDepthInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = None
val_dataloader = None
val_evaluator = None
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
vis_every_iters = 100
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'Outputs/train/depth_general/sapiens_1b_depth_general-1024x768/node/10-17-2024_12:55:06'
