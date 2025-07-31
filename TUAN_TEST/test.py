# import pandas as pd
# import re
# import openpyxl

# # Đọc file log
# with open('filter_log.txt', 'r') as f:
#     lines = f.readlines()

# # Danh sách lưu kết quả
# results = []

# # Biến tạm để gom 4 dòng/epoch
# epoch_data = {}

# for line in lines:
#     if 'TRAIN  loss' in line:
#         match = re.search(r'loss: ([\d.]+) acc: ([\d.]+)', line)
#         if match:
#             epoch_data['train_loss'] = float(match.group(1))
#             epoch_data['train_acc'] = float(match.group(2))
    
#     elif 'AVG TRAIN time' in line:
#         match = re.search(r'sample \(sec\): ([\d.]+)', line)
#         if match:
#             epoch_data['train_time_per_sample'] = float(match.group(1))

#     elif 'VALIDATION  acc:' in line and 'Top' not in line:
#         match = re.search(r'acc: ([\d.]+)', line)
#         if match:
#             epoch_data['val_acc'] = float(match.group(1))

#     elif 'VALIDATION  Top 5 acc:' in line:
#         match = re.search(r'Top 5 acc: ([\d.]+)', line)
#         if match:
#             epoch_data['val_top5_acc'] = float(match.group(1))
#             # Đã gom đủ 4 dòng -> thêm vào kết quả
#             results.append(epoch_data)
#             epoch_data = {}

# # Ghi vào Excel
# df = pd.DataFrame(results)
# df.index.name = 'epoch'
# df.to_excel('training_log_summary.xlsx', engine='openpyxl')

# print("Đã ghi xong vào training_log_summary.xlsx")
import pandas as pd

# Đọc file
with open("requirements.txt", "r") as f:
    lines = f.read().splitlines()

# Tách tên và version
packages = []
for line in lines:
    if '=' in line:
        name, version = line.strip().split('=', 1)
    else:
        name = line.strip()
        version = ''  # Không có phiên bản
    packages.append((name, version))

# Tạo DataFrame
df = pd.DataFrame(packages, columns=["Package", "Version"])

# Xuất ra Excel
df.to_excel("requirements.xlsx", index=False, engine="openpyxl")

print("✅ Đã xuất file packages_list.xlsx thành công.")
