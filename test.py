from siformer.model import SiFormer, SpoTer
import torch

# Tạo model
model = SiFormer(num_classes=100)

# Tạo dữ liệu mẫu
l_hand = torch.randn(2, 204, 21, 2)  # 2 samples, 204 frames, 21 keypoints, x,y
r_hand = torch.randn(2, 204, 21, 2)
body = torch.randn(2, 204, 12, 2)

# Gọi model - PyTorch tự động gọi forward()
predictions = model(l_hand, r_hand, body, training=True)

print(predictions.shape)  # Output: torch.Size([2, 100])
# 2 samples, mỗi sample có 100 điểm số cho 100 classes

print("###############")

print(predictions)

print("###############")
