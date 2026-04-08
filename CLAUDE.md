# Training WAFT Stereo
# Training depth model
Tôi đang cố gắng huấn luyện một mô hình depth mới sử dụng dữ liệu của tôi, tham khảo từ thư viện OpenStereo (tìm thấy trong đường dẫn OpenStereo) để kế thừa pipeline huấn luyện. 

## Chuẩn bị dữ liệu
Dữ liệu tôi đã chuẩn bị có thể tìm thấy trong `/home/thetrung/Documents/CT_UAV/Obstacle_Avoidance_ws3/data/stereo-smallbaseline`. Dữ liệu có thể load vào dataset và dataloader thông qua `bridgedepth/dataloader/depth_dataset.py`. 

## Huấn luyện mô hình
Mô hình sẽ được tôi custom lại trong `WAFT-Stereo`. Huấn luyện và đánh giá dựa trên các metric trong `reference/evaluation/evaluation.py`.

## Mục tiêu 1 (Hoàn thành trước chạy trên máy tính)
Hãy giúp tôi thực hiện:
1. update lại code training và evaluation để phù hợp với dữ liệu của tôi. Sau mỗi 5 epoch huấn luyện, hãy đánh giá mô hình trên tập validation và lưu lại mô hình checkpoint. Quá trình huấn luyện sẽ được quản lý qua tensorboard.
2. Update main.py để huấn luyện mô hình.
3. Tạo file train.sh với các lệnh/ parameters cần thiết để chạy huấn luyện. Có thể load pretrained model nếu có checkpoint trước đó, hoặc bắt đầu huấn luyện từ đầu nếu không có checkpoint nào.
4. Quá trình debug sẽ được thực hiện trên máy tính của tôi, vì vậy bạn cần đảm bảo rằng code có thể chạy được trên máy tính của tôi với cấu hình phần cứng hiện tại (GPU, RAM, v.v.). Bạn có thể sử dụng logging, tqdm để theo dõi quá trình huấn luyện và đánh giá, giúp dễ dàng phát hiện và xử lý các vấn đề phát sinh trong quá trình này. Và môi trường cần được cài đặt trên anaconda, named `waft`.
5. khi evaluation mỗi 5 epoch sẽ inference dữ liệu trong `test_images` và lưu kết quả trong tensorboard để có thể so sánh giữa các checkpoint và các mô hình khác trong tương lai. Lưu ý: Cần calib stereo camera trước khi inference để đảm bảo kết quả đánh giá chính xác.

## Mục tiêu 2 (Hoàn thiện trên kaggle)
Sau khi hoàn thành mục tiêu 1, tiếp theo tôi cần triển khai code trên kaggle để có thể huấn luyện mô hình trên đó. Tôi đã tạo sẵn 2 kernel cho bạn, một kernel là utility, một kernel là môi trường training. Môi trường training sẽ không có quyền truy cập internet, vì vậy bạn cần đảm bảo rằng tất cả các thư viện cần thiết đã được cài đặt trong môi trường utility và sau đó chuyển chúng sang môi trường training. Không thay đổi id hay tên trong metadata của kernel để đảm bảo rằng chúng có thể truy cập được với kaggle server.

1. Cài đặt môi trường trên kaggle để có thể chạy được code của tôi, tham khảo thư mục `kaggle` gồm 2 môi trường `kaggle/training` chưa notebook để huấn luyện và `kaggle/utility` chứa các file cần thiết để cài đặt môi trường (lưu ý môi trường training không access được internet).

2. Tiến hành huấn luyện và note lại quá trình huấn luyện mô hình ( stt, date, name, specification, result from type of metrics ... ) để có thể so sánh với các mô hình khác trong tương lai.

3. Các thông số huấn luyện có thể được điều chỉnh trên kaggle thông qua các lệnh chạy trong python notebook. 

4. Tạo file csv để lưu lại kết quả huấn luyện của mô hình, bao gồm tên, kết quả đánh giá để có thể so sánh với các mô hình khác trong tương lai.

**Một số lưu ý khi edit notebook training:**
- UTILITY_PATH = "/kaggle/input/notebooks/laithetrung/waft-utility-script"
- CODE_SRC: "/kaggle/input/notebooks/laithetrung/waft-repo/JeDepth/train.sh"
- DATASET_SRC = "/kaggle/input/datasets/laithetrung/stereo-smallbaseline"

## Quy ước code:
- Code sẽ được tổ chức thành các hàm và lớp rõ ràng, có chú thích đầy đủ để giải thích mục đích và cách sử dụng của từng phần.
- Sử dụng logging, tqdm, visualization và tensorboard để theo dõi quá trình huấn luyện và đánh giá, giúp dễ dàng phát hiện và xử lý các vấn đề phát sinh trong quá trình này.
- Trước khi bắt đầu suy luận thì hãy tóm tắt các function của mỗi file, class và mối liên hệ của nó với các file khác trong source và lưu vào file md. Dúng nó trong mọi quá trình suy luận để hạn chế ảo tưởng.
- Trước khi code thì hãy nêu plan của bạn, nếu plan chia làm nhiều bước, hãy đề xuất thứ tự thực hiện các bước và thực hiẹn từng bước 1 để tôi có thể quản lý.