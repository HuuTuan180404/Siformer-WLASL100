import subprocess
import sys
import pandas as pd

# Lấy version Python hiện tại
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Lấy danh sách gói từ pip
result = subprocess.run(['pip', 'list', '--format=freeze'], stdout=subprocess.PIPE, text=True)

# Phân tích kết quả
lines = result.stdout.strip().split('\n')
packages = [line.split('==') for line in lines if '==' in line]

# Thêm Python version cho từng dòng
packages_with_version = [pkg + [python_version] for pkg in packages]

# Tạo DataFrame
df = pd.DataFrame(packages_with_version, columns=['Package', 'Version', 'Python Version'])

# Ghi ra file Excel
df.to_excel('pip_packages_with_python_version.xlsx', index=False, engine='openpyxl')

print("✅ Đã lưu pip list kèm Python version vào pip_packages_with_python_version.xlsx")
