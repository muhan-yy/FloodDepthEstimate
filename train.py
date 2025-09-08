from ultralytics import YOLO

# Load a model
# model = YOLO("checkpoints/segmentation/yolo11n-seg.pt")  # Load a pretrained model (recommended for training)
# 完成的类别   car   motorcycle  electricCar people   wheel     licensePlate

conferenceType = 'all6'  # all6
modelType = 'n'

"""
# 模块参数量增加比较

ori : 2.75G
YOLO11n-seg-all6-ori summary: 355 layers, 3,760,194 parameters, 3,760,178 gradients, 53.0 GFLOPs

head_DAT : 加2-3个地方都加会超过CUDA 加1个             更好地聚焦于相关区域
YOLO11n-seg-all6-head_DAT summary: 369 layers, 4,047,938 parameters, 4,047,922 gradients, 53.3 GFLOPs


backbone_C3k2_HTB                                    适用于噪声大、图像质量地的任务                   提高恶劣天气图像恢复的性能
YOLO11n-seg-all6-backbone_C3k2_HTB summary: 369 layers, 5,137,606 parameters, 5,137,590 gradients, 60.8 GFLOPs

backbone_C2PSA_DAT                                   更好地聚焦于相关区域                            修改注意力机制，更聚焦
YOLO11n-seg-all6-backbone_C2PSA_DAT summary: 357 layers, 3,787,202 parameters, 3,787,186 gradients, 53.1 GFLOPs


head_Segment_dyhead                                  动态卷积，增加参数量，但是不增加FLOP             参照物形状差别大
YOLO11n-seg-all6-head_Segment_dyhead summary: 356 layers, 3,557,210 parameters, 3,557,194 gradients, 51.6 GFLOPs


head_MLLABlock      有bug RuntimeError: "fill_empty_deterministic_" not implemented for 'ComplexHalf'  可能是CUDA版本问题
YOLO11n-seg-all6-head_MLLABlock summary: 418 layers, 4,842,498 parameters, 4,842,482 gradients, 55.2 GFLOPs
"""

# DCNV3   ori  head_DAT   backbone_C2PSA_DAT   head_Segment_dyhead   backbone_C3k2_HTB  head_MLLABlock   neck_GSConv   neck_BiFPN
model = YOLO(f"yamls/yolo11{modelType}-seg-{conferenceType}-neck_BiFPN.yaml") 
# Train the model
model.train(data=f"yamls/mulReference-seg-{conferenceType}.yaml", 
            epochs=300, 
            imgsz=640, 
            workers=0,
            batch=2, # water 32
            device='cuda', 
            lr0=1e-3, 
            patience=10, 
            optimizer='AdamW',
            seed=1105,
            cos_lr=True)
