# Data preprocessing and machine learning projects - Real estate price prediction

## Giới thiệu

Dự án này trình bày toàn bộ quy trình xây dựng hệ thống dữ liệu và mô hình Machine Learning — từ thu thập dữ liệu, tiền xử lý, gọi API của mô hình ngôn ngữ lớn (LLM) trích xuất văn bản mô tả đối tượng và huấn luyện mô hình máy học.

Repository được tổ chức theo chuẩn dự án ML thực tế, dễ mở rộng và tái sử dụng.

## 📂 Cấu trúc thư mục

├── data/                      # Dữ liệu thô & dữ liệu đã xử lý
├── scraping-data/             # Code thu thập dữ liệu (scraping/crawling)
├── preprocessing-data/        # Làm sạch, transform, EDA
├── model/                     # Huấn luyện mô hình, đánh giá, lưu model
└── gemini-api-calling/        # Gọi API Gemini / LLM phục vụ inference

### 📁 data/

Gồm các file csv chứa links tới bất động sản, dữ liệu thô, dữ liệu đã qua xử lý sơ bộ và dữ liệu đã được tiền xử lý

### 📁 scraping-data/

Chứa script tự động thu thập dữ liệu từ web batdongsan.com.vn

- Công nghệ: requests, BeautifulSoup, Selenium

- Output lưu vào thư mục data/

### 📁 gemini-api-calling/

Gọi API đến mô hình Gemini Flash 1.5 (Free Tier), cấu trúc format output phù hợp để đưa vào 1 thuộc tính.

### 📁 preprocessing-data/

Làm sạch dữ liệu: 
- Chuẩn hóa về các giá trị thống nhất (địa chỉ, đơn vị...)
- Xử lý văn bản, chuỗi để trích xuất thông tin hữu ích
- Xử lý null, duplicates, outliers, noises

Phân tích dữ liệu:
- Phân tích đơn biến, đa biến
- Thực hiện các kiểm định để biết tương quan, phụ thuộc giữa các biến

Feature engineering:
- Thêm các thuộc tính giúp mô hình ML học tốt hơn

Chuẩn hóa dữ liệu dạng số:
- Từ phân phối của dữ liệu (xem ở bước phân tích đơn biến), chọn Scaler phù hợp với từng thuộc tính: đa số thuộc tính số có phân phối lệch phải do có nhiều ngoại lai, độ lệch và độ nhọn rất cao, nên dùng biến đổi log để giảm độ lệch và giảm ảnh hưởng của ngoại lệ. Các thuộc tính có phân phối gần chuẩn thì dùng chuẩn hóa Z-score và MinMax.

Feature Selection:
- Kiểm tra VIF, ý nghĩa các biến và loại bỏ các biến đầu vào có tương quan cao với nhau, tránh hiện tượng đa cộng tuyến

Tạo dataset cuối cho training model
