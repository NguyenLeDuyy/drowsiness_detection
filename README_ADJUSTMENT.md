# Báo cáo điều chỉnh và cải tiến hệ thống phát hiện buồn ngủ

## 1. Vấn đề ban đầu
- Mô hình dự đoán sai khi chạy thực tế dù test.py cho kết quả tốt.
- Nguyên nhân: Không nhất quán trong chuẩn hóa dữ liệu giữa train, test và dự đoán thực tế.

## 2. Các thay đổi đã thực hiện

### a. Lưu lại thống kê chuẩn hóa
- Trong `train.py`, sau khi adapt Normalization layer, đã lưu lại `norm_mean.npy` và `norm_var.npy`.
- Tính `norm_std = sqrt(norm_var + 1e-8)` để chuẩn hóa.

### b. Tái sử dụng mean, var, std ở test.py và drowsiness-detection.py
- Load lại `norm_mean` và `norm_var` trong các file này.
- Khi tiền xử lý ảnh, luôn dùng:  
  `(image - norm_mean) / norm_std`

### c. Sửa các hàm preprocess
- Không gọi `.adapt()` trên từng ảnh khi test hoặc dự đoán thực tế.
- Đảm bảo shape đầu vào đúng (1, 224, 224, 3).

### d. Kiểm tra lại toàn bộ pipeline
- Đảm bảo test.py và drowsiness-detection.py đều cho kết quả tốt, nhất quán.

## 3. Kết quả đạt được
- Mô hình dự đoán ổn định, chính xác hơn khi chạy thực tế.
- Không còn hiện tượng dự đoán sai do chuẩn hóa không nhất quán.
- Đầu ra của test.py và drowsiness-detection.py gần như giống nhau về độ chính xác.

## 4. Một số hình ảnh minh họa lỗi dự đoán
- Đã sử dụng script `analyze_incorrect.py` để trực quan hóa các trường hợp dự đoán sai, giúp phân tích nguyên nhân và cải thiện mô hình.

## 5. Bài học rút ra
- Chuẩn hóa dữ liệu nhất quán giữa train, test, inference là cực kỳ quan trọng.
- Luôn lưu lại các tham số chuẩn hóa từ tập huấn luyện và tái sử dụng ở mọi nơi.
- Kiểm tra kỹ shape của dữ liệu trước khi đưa vào model để tránh lỗi.

---

*File này giúp tổng hợp quá trình cải tiến, thuận tiện cho việc báo cáo, bảo trì hoặc chia sẻ với người khác.*