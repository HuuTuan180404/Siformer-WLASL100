import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

def split_csv_to_two(input_csv, output_file1, output_file2, split_ratio, stratify_column, seed):
    """
    Tách một file CSV thành hai file dựa trên một tỷ lệ cho trước,
    có hỗ trợ tách ng-ẫu nhiên và tách có phân tầng.
    """
    print("--- Bắt đầu quá trình tách file ---")

    # --- Bước 1: Đọc file CSV gốc ---
    try:
        df = pd.read_csv(input_csv)
        print(f"Đọc thành công file '{input_csv}' với {len(df)} hàng.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{input_csv}'")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    # --- Bước 2: Kiểm tra các tham số ---
    if not (0 < split_ratio < 1):
        print(f"Lỗi: Tỷ lệ tách (split_ratio) phải nằm trong khoảng (0, 1). Giá trị hiện tại: {split_ratio}")
        return

    if stratify_column and stratify_column not in df.columns:
        print(f"Lỗi: Cột phân tầng '{stratify_column}' không tồn tại trong file CSV.")
        print(f"Các cột có sẵn là: {list(df.columns)}")
        return

    # --- Bước 3: Thực hiện việc tách ---
    test_size = 1.0 - split_ratio

    try:
        df1, df2 = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=df[stratify_column] if stratify_column else None
        )
        if stratify_column:
            print(f"Đã thực hiện tách có phân tầng dựa trên cột '{stratify_column}'.")
        else:
            print("Đã thực hiện tách ngẫu nhiên (không phân tầng).")
            
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình tách dữ liệu: {e}")
        return

    # --- Bước 4: In thông tin và ghi ra file ---
    print(f"\nFile 1 sẽ có {len(df1)} hàng ({split_ratio:.0%}).")
    print(f"File 2 sẽ có {len(df2)} hàng ({test_size:.0%}).")

    try:
        # Tạo thư mục nếu cần
        # os.makedirs(os.path.dirname(output_file1), exist_ok=True)
        # os.makedirs(os.path.dirname(output_file2), exist_ok=True)
        
        # Ghi file
        df1.to_csv(output_file1, index=False, encoding='utf-8-sig')
        print(f"-> Đã lưu file 1 vào: '{output_file1}'")
        
        df2.to_csv(output_file2, index=False, encoding='utf-8-sig')
        print(f"-> Đã lưu file 2 vào: '{output_file2}'")
    except Exception as e:
        print(f"Lỗi khi ghi file output: {e}")

    print("\n--- Quá trình hoàn tất! ---")

def main():    
    split_csv_to_two(
        input_csv='WLASL100_val_25fps.csv',
        output_file1='val_80.csv',
        output_file2='val_20.csv',
        split_ratio=0.375,
        stratify_column='labels',
        seed=42
    )

if __name__ == "__main__":
    main()