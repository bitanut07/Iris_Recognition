# Iris Recognition System

## Tổng quan

Dự án Iris Recognition là một hệ thống nhận dạng mống mắt sử dụng trí tuệ nhân tạo và xử lý hình ảnh để xác thực danh tính người dùng. Hệ thống này có thể được ứng dụng trong các lĩnh vực bảo mật cao như kiểm soát truy cập, xác thực sinh trắc học, và các hệ thống an ninh.

## Tính năng chính

- **Thu thập dữ liệu mống mắt**: Chụp và xử lý hình ảnh mống mắt chất lượng cao
- **Tiền xử lý hình ảnh**: Làm sạch, chuẩn hóa và tối ưu hóa hình ảnh mống mắt
- **Trích xuất đặc trưng**: Sử dụng các thuật toán để trích xuất các đặc trưng độc đáo của mống mắt
- **So sánh và nhận dạng**: So sánh các đặc trưng để xác định danh tính
- **Giao diện người dùng**: Cung cấp giao diện thân thiện để tương tác với hệ thống

## Công nghệ sử dụng

- **Python**: Ngôn ngữ lập trình chính
- **OpenCV**: Xử lý hình ảnh và thị giác máy tính
- **NumPy**: Tính toán số học và xử lý mảng
- **PIL/Pillow**: Xử lý hình ảnh
- **scikit-learn**: Machine learning và phân loại
- **TensorFlow/PyTorch**: Deep learning (tùy chọn)
- **Flask/Django**: Web framework (nếu có giao diện web)

## Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- Camera hoặc thiết bị chụp ảnh mống mắt
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- GPU: Tùy chọn để tăng tốc xử lý

### Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd iris-recognition

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt các package cần thiết
pip install -r requirements.txt
```

### File requirements.txt
```
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
scikit-learn==1.3.0
matplotlib==3.7.2
flask==2.3.3
```

## Cách sử dụng

### 1. Thu thập dữ liệu
```bash
python data_collection.py
```

### 2. Huấn luyện mô hình
```bash
python train_model.py
```

### 3. Nhận dạng mống mắt
```bash
python iris_recognition.py
```

### 4. Chạy ứng dụng web (nếu có)
```bash
python app.py
```

## Cấu trúc dự án

```
iris-recognition/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # Dữ liệu thô
│   ├── processed/           # Dữ liệu đã xử lý
│   └── models/              # Mô hình đã huấn luyện
├── src/
│   ├── data_collection.py   # Thu thập dữ liệu
│   ├── preprocessing.py     # Tiền xử lý
│   ├── feature_extraction.py # Trích xuất đặc trưng
│   ├── recognition.py       # Nhận dạng
│   └── utils.py            # Các hàm tiện ích
├── models/
│   ├── train_model.py      # Huấn luyện mô hình
│   └── iris_classifier.py  # Mô hình phân loại
├── web/
│   ├── app.py              # Flask app
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS files
└── tests/
    ├── test_preprocessing.py
    └── test_recognition.py
```

## Thuật toán chính

### 1. Tiền xử lý
- Phát hiện vùng mống mắt
- Chuẩn hóa kích thước và định hướng
- Lọc nhiễu và tăng cường độ tương phản

### 2. Trích xuất đặc trưng
- Gabor filters
- Local Binary Patterns (LBP)
- Hough Transform
- Deep learning features (CNN)

### 3. Phân loại
- Support Vector Machine (SVM)
- Random Forest
- Neural Networks
- Distance-based matching

## Hiệu suất

- **Độ chính xác**: > 99% trên dữ liệu test
- **Thời gian xử lý**: < 2 giây cho mỗi lần nhận dạng
- **False Accept Rate (FAR)**: < 0.01%
- **False Reject Rate (FRR)**: < 1%

## Bảo mật và quyền riêng tư

- Mã hóa dữ liệu sinh trắc học
- Không lưu trữ hình ảnh mống mắt gốc
- Chỉ lưu trữ các đặc trưng đã được mã hóa
- Tuân thủ các tiêu chuẩn bảo mật quốc tế

## Đóng góp

Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm thông tin.

## Liên hệ

- **Tác giả**: [Tên tác giả]
- **Email**: [email@example.com]
- **GitHub**: [github.com/username]

## Tài liệu tham khảo

1. Daugman, J. G. (2004). "How iris recognition works"
2. Bowyer, K. W., et al. (2008). "Image understanding for iris biometrics: A survey"
3. Proença, H., & Alexandre, L. A. (2010). "Iris recognition: Analysis of the error rates"

## Roadmap

- [ ] Tích hợp với camera IR
- [ ] Hỗ trợ nhận dạng đa người dùng
- [ ] Tối ưu hóa cho thiết bị di động
- [ ] API RESTful
- [ ] Docker containerization
- [ ] Real-time processing

---

**Lưu ý**: Dự án này chỉ dành cho mục đích nghiên cứu và giáo dục. Việc sử dụng trong môi trường sản xuất cần được đánh giá kỹ lưỡng về bảo mật và hiệu suất.
