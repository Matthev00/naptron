import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

output_path = "work_dirs/faster_rcnn_r50_fpn_1x_voc0712_cocofmt/latest.pth"

torch.save(model.state_dict(), output_path)
print(f"Model zapisany jako: {output_path}")
