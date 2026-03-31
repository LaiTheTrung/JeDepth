# EDA and Data Prepocessing for Depth Estimation Dataset

## 1. Mục tiêu
- Thực hiện EDA cho các bộ dataset depth estimation: `fat_indoor`, `fat_outdoor`, `Stereo2k_indoor`, và `spring_outdoor`. EDA sẽ tập trung vào việc phân tích phạm vi giá trị của depth/disparity, tỷ lệ phần trăm pixel có giá trị hợp lệ, và phân bố của các giá trị này. Ngoài ra cũng sẽ đếm số lượng ảnh thuộc indoor vs outdoor.
- Chuẩn bị dữ liệu cho việc huấn luyện mô hình depth estimation, bao gồm việc chuyển đổi depth maps sang disparity maps nếu cần thiết và thống kê chúng thành một file CSV để dễ dàng truy cập trong quá trình huấn luyện. CSV sẽ bao gồm các cột: `image_path`, `depth_path`, `disparity_path`, `min_depth`, `max_depth`, `environment`. 
## 2. Cấu trúc dataset:
- `fat_indoor/fat_outdoor` → **depth** Giá trị depth được lưu dưới dạng 16-bit PNG, cần được chuyển đổi sang float32 và chia với 10000.0 để có được giá trị depth thực tế tính bằng mét. (baseline = 6 cm, FL = 768.2 pixel)
- `Stereo2k_indoor` → **disparity** Giá trị disparity được lưu dưới dạng 16-bit PNG, cần được chuyển đổi sang float32 và chia với 100.0 để có được giá trị disparity thực tế tính bằng pixel. (baseline = 10cm)
- `spring_outdoor` -> **disparity** Giá trị disparity được lưu dưới dạng file .dsp5 (HDF5), cần được đọc bằng thư viện h5py để trích xuất mảng disparity thực tế. (baseline = 6.5 cm)
## 3. Data Preprocessing
- Sau khi đã thống kê các file csv. Hãy viết một script để chuyển đổi tất cả các depth maps sang disparity maps (nếu cần thiết) và lưu chúng vào một thư mục mới. Dữ liệu mới sẽ được lọc dựa trên (`min_depth`, `max_depth` , `environment`) và resize thành new_size =(640x480). Đồng thời cập nhật file CSV với đường dẫn mới đến các disparity maps đã được chuyển đổi. Công thức chuyển đổi từ depth (d) sang disparity (disp) sẽ là: `disp = (f * B) / d`, trong đó f là tiêu cự của camera và B là baseline giữa hai camera stereo, giá trị disp cũng sẽ được điều chỉnh theo kích thước ảnh mới nếu cần thiết (disp = disp*resize_ratio). Dữ liệu disp sẽ được lưu dưới dạng 16-bit PNG (bằng cách nhân giá trị disparity thực tế với 100.0 và chuyển sang uint16) để tiết kiệm dung lượng lưu trữ.
- Có thể config được các tham số gồm: `new_size`, `min_depth`, `max_depth`, `environment` để lọc dữ liệu và resize ảnh. Ví dụ: chỉ giữ lại các ảnh indoor có depth trong khoảng 0.5m đến 10m, và resize chúng về kích thước 640x480. 
- Thiết kế dataloader, dataset để load dữ liệu từ file CSV đã được chuẩn bị, bao gồm việc đọc các ảnh và disparity maps, cũng như áp dụng các phép biến đổi cần thiết như normalization, augmentation (nếu cần) trong quá trình huấn luyện mô hình depth estimation. Dataloader sẽ hỗ trợ việc batching và shuffling dữ liệu để tăng hiệu quả huấn luyện.
# 4. Language and Frameworks:
- Sử dụng Python làm ngôn ngữ lập trình chính.
- Sử dụng các thư viện phổ biến như NumPy, Pandas, OpenCV, PIL, h5py để xử lý dữ liệu và thực hiện EDA.
- Sử dụng Pytorch để xây dựng dataloader và dataset cho việc huấn luyện mô hình depth estimation.
# 5. Quy ước code:
- Code sẽ được tổ chức thành các hàm và lớp rõ ràng, có chú thích đầy đủ để giải thích mục đích và cách sử dụng của từng phần.
- Sử dụng logging, visualize để theo dõi quá trình EDA và data preprocessing, giúp dễ dàng phát hiện và xử lý các vấn đề phát sinh trong quá trình này.
- Tất cả các file CSV và dữ liệu đã được chuyển đổi sẽ được lưu trữ một cách có tổ chức trong thư mục `processed_data`, với cấu trúc rõ ràng để dễ dàng truy cập và sử dụng trong quá trình huấn luyện mô hình depth estimation.

# 