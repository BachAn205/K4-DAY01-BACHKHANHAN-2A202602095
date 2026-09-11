# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 13-9-2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không (sử dụng luồng chuẩn và thông số mặc định của bài thực hành)

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}` (thuộc sample `traffic`, coco_image_id: 210273).
- Record này mô tả toàn ảnh như thế nào? 
Record này gán một nhãn phân loại duy nhất ở cấp độ toàn bộ bức ảnh (bức ảnh được dự đoán thuộc lớp xe taxi - "cab" với độ tin cậy ~51.09%). Nó không cung cấp thông tin vị trí tọa độ, kích thước hay phân biệt từng cá thể/vật thể riêng lẻ xuất hiện trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? 
Nhóm tác giả / tổ chức xây dựng bộ dữ liệu và taxonomy chuẩn (cụ thể ở đây là tập dữ liệu ImageNet-1K gồm 1,000 danh mục lớp theo WordNet synsets) khi huấn luyện checkpoint `yolo11n-cls.pt`.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id`: Cần thiết để máy tính lập chỉ mục (index) và xử lý dữ liệu nhất quán trong pipeline/codebase.
  - `class_name`: Giúp con người (annotator, reviewer, engineer) dễ đọc và hiểu trực tiếp ý nghĩa ngữ nghĩa của nhãn.
  - `taxonomy_name`: Định danh không gian nhãn và quy ước ngữ nghĩa cụ thể (ví dụ ImageNet-1K khác COCO-80), tránh xung đột khi cùng một tên lớp hoặc cùng ID có định nghĩa khác nhau ở các bộ dữ liệu khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần quy định rõ nguyên tắc chọn nhãn: ưu tiên chủ thể chiếm diện tích lớn nhất (dominant subject), chủ thể ở vị trí trung tâm, hoặc thứ tự ưu tiên lớp theo yêu cầu nghiệp vụ; hoặc nếu chuyển sang bài toán multi-label classification thì phải quy định rõ tiêu chuẩn đánh dấu nhiều nhãn đồng thời.
- Vì sao model score không phải ground truth? Model score (confidence score) chỉ là xác suất thống kê do mô hình ước lượng dựa trên các trọng số đã học từ dữ liệu quá khứ (vẫn có thể chứa sai lệch, bias hoặc hallucination). Ground truth là sự thật khách quan đã được con người (human annotator / domain expert) kiểm chứng và xác nhận chuẩn xác theo guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}` (thuộc sample `kitchen`, threshold 0.35).
- Diễn giải vị trí box bằng lời: Vật thể là một người đứng ở phía bên phải khung hình, có góc trên bên trái box tại tọa độ (x_min = 385.33 px, y_min = 69.24 px) và góc dưới bên phải tại (x_max = 498.92 px, y_max = 348.92 px) trên tổng thể bức ảnh kích thước 640x427 px; chiều rộng box là 113.58 px và chiều cao box là 279.68 px (bao phủ toàn bộ thân người từ đầu đến chân).
- So sánh số prediction ở hai threshold:
  - Ở threshold mặc định 0.35: Model trả về **11 predictions** cho sample `kitchen` (gồm person, oven, bowl, cup).
  - Nếu nâng threshold lên 0.50: Số predictions giảm xuống còn **6 predictions** (chỉ giữ lại các vật thể có độ tự tin cao như person, oven, bowl lớn rõ ràng).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi hạ threshold: Độ bao phủ (recall) tăng, nhận diện được nhiều vật thể nhỏ/mờ hơn nhưng reviewer phải tốn nhiều thời gian kiểm tra và loại bỏ các dự đoán sai/rác (false positives).
  - Khi nâng threshold: Khối lượng reviewer cần xem giảm đi (chỉ lọc các box có độ tin cậy cao), nhưng nguy cơ bỏ sót các vật thể thực tế trong ảnh (false negatives / sụt giảm recall) tăng lên.
- Đề xuất một quy tắc box chặt: Bounding box phải bao bọc sát khít các cạnh biên ngoài cùng nhìn thấy được của vật thể (tight bounding box), khoảng trống thừa không vượt quá 2-3 pixel và tuyệt đối không được cắt lẹm vào bất kỳ bộ phận nhìn thấy nào của vật thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định rõ: vẽ box bao phủ toàn bộ phần nhìn thấy được (visible part) hay ước lượng toàn bộ hình thể (full extent); quy định tỷ lệ diện tích nhìn thấy tối thiểu (ví dụ >10-20%) để được gán nhãn. Khi vật thể bị cắt mép hoặc che khuất quá mức không thể nhận dạng chắc chắn, annotator cần escalate cho Lead/QA quyết định thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `{"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318, "polygon_point_count": 348, "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], ...]}` (thuộc sample `kitchen`, checkpoint `yolo11n-seg.pt`).
- Polygon bổ sung chi tiết gì so với box? Polygon cung cấp chi tiết hình dạng hình học chính xác đến từng pixel theo đường viền thực tế (silhouette/contour) của đối tượng, loại bỏ hoàn toàn các pixel nền trống và vật thể xung quanh mà bounding box hình chữ nhật không thể phân tách được.
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` dùng để định danh và phân biệt từng cá thể/thực thể độc lập của cùng một lớp đối tượng trong cùng một bức ảnh (ví dụ phân biệt người `kitchen-001` với người `kitchen-009`, hoặc giữa các cái bát khác nhau). Nó **không phải** là `class_id` (mã danh mục lớp) và **không phải** là `tracking_id` (mã theo dõi vật thể xuyên suốt các frame video).
- Đề xuất một quy tắc biên mask: Đường bao mask polygon phải ôm khít đường viền thực tế của vật thể với sai số không quá 1-2 pixel; không được chừa pixel nền vào trong mask và không được cắt lẹm vào các chi tiết/bộ phận nhô ra của đối tượng (như cánh tay, bàn tay, quai nồi, vành bát).
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Khi hai vật thể tiếp xúc hoặc đè lên nhau (ví dụ người cầm bát, cốc đặt trên bàn), guideline cần quy định rõ ranh giới tiếp xúc thuộc về instance nào và tách thành các polygon riêng biệt. Đối với các vùng biên mờ do chuyển động (motion blur) hoặc độ sâu trường ảnh (DOF), guideline cần quy định lấy biên theo đường gradient tương phản rõ nhất hoặc escalate cho QA/Lead nếu ranh giới bị nhập nhằng.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp ảnh (`class_id`, `class_name` thuộc taxonomy chuẩn) | Ảnh có nhiều chủ thể đồng thời (vừa có taxi, xe bus, xe con), khó xác định nhãn duy nhất | Chọn 1 nhãn của chủ thể nổi bật nhất (dominant) theo đúng tiêu chí ưu tiên trong guideline | Kiểm tra xem nhãn được chọn có đúng với chủ thể chính theo quy định guideline hay không |
| Phát hiện vật thể | Danh sách Bounding Box (`[x_min, y_min, x_max, y_max]` + `class_id`) | Box bị lỏng (chứa nhiều nền) hoặc bỏ sót các vật thể nhỏ/nằm ở vùng khuất (như bát/cốc nhỏ) | Vẽ box bao sát từng vật thể nhìn thấy, quét toàn ảnh để không bỏ sót đối tượng | Kiểm tra độ khít của box (IoU), phát hiện box thừa (false positive) hoặc sót vật thể (false negative) |
| Instance segmentation | Danh sách Polygon (`[[x, y], ...]` + `instance_id` + `class_id`) | Ranh giới giữa các vật thể tiếp xúc bị dính liền, hoặc đường biên mask lấn vào nền | Cắt tỉa từng điểm polygon bám sát đường viền pixel của từng cá thể riêng biệt | Zoom-in kiểm tra độ chính xác đường biên (boundary alignment), độ tách biệt instance và độ phủ pixel |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Tuyệt đối không đưa dữ liệu nhạy cảm, thông tin định danh cá nhân (PII - họ tên, CCCD, khuôn mặt chưa làm mờ, biển số xe riêng tư) hoặc dữ liệu nội bộ/bí mật kinh doanh lên môi trường công khai/Colab chưa được cấp phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Giảng viên / Project Lead / Quản lý dữ liệu (Mentor / Data Manager) phụ trách để nhận chỉ dẫn xử lý và thực hiện đúng quy trình escalation.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
