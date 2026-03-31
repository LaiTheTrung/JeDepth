# Training depth model
Tôi đang cố gắng huấn luyện một mô hình depht mới sử dụng dữ liệu của tôi, tham khảo từ thư viện OpenStereo (tìm thấy trong đường dẫn OpenStereo) để kế thừa pipeline huấn luyện. 

## Chuẩn bị dữ liệu
Dữ liệu tôi đã chuẩn bị có thể tìm hấy trong `data/processed_data`. Dữ liệu có thể load vào dataset và dataloader thông qua `jedepth/dataset/depth_dataset.py`. 

## Huấn luyện mô hình
Mô hình sẽ được tôi custom lại trong `jedepth`. Huấn luyện và đánh giá dựa trên các metric trong `jedepth/evaluation/evaluation.py`.

## Mục tiêu
Hãy giúp tôi thực hiện:
1. update lại code training và evaluation để phù hợp với dữ liệu của tôi. Sau mỗi 5 epoch huấn luyện, hãy đánh giá mô hình trên tập validation và lưu lại mô hình checkpoint. Quá trình huấn luyện sẽ được quản lý qua tensorboard.
2. Tạo file train.py để huấn luyện mô hình.
3. Tạo file train.sh với các lệnh/ parameters cần thiết để chạy huấn luyện.

## Quy ước code:
- Code sẽ được tổ chức thành các hàm và lớp rõ ràng, có chú thích đầy đủ để giải thích mục đích và cách sử dụng của từng phần.
- Sử dụng logging, tqdm, visualization và tensorboard để theo dõi quá trình huấn luyện và đánh giá, giúp dễ dàng phát hiện và xử lý các vấn đề phát sinh trong quá trình này.
