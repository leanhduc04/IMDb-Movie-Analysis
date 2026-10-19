# 🎬 IMDb Movie Analysis & Dashboard

## 1. Giới thiệu

Dự án sử dụng Python để phân tích bộ dữ liệu **1.000 bộ phim IMDb giai đoạn 2006–2016**, trực quan hóa kết quả và xây dựng Dashboard bằng Streamlit. Dự án cũng thử nghiệm **Linear Regression** và **Random Forest Regression** để ước tính doanh thu phim.

## 2. Chức năng chính

- **Tiền xử lý dữ liệu:** kiểm tra dữ liệu thiếu, chuẩn hóa tên cột và thông tin thể loại/diễn viên.
- **Phân tích khám phá dữ liệu (EDA):** phân bố các biến định lượng, xu hướng theo năm, phân tích thể loại, bảng xếp hạng phim/đạo diễn và tương quan Pearson.
- **Học máy:** huấn luyện, đánh giá và so sánh hai mô hình hồi quy dự đoán doanh thu.
- **Dashboard Streamlit:** hiển thị biểu đồ tương tác theo bộ lọc, tra cứu phim, xem bảng dữ liệu và ước tính doanh thu.

## 3. Cấu trúc thư mục

```text
Movie_Data_Analysis/
├── data/
│   ├── imdb_movie_dataset.csv       # Dữ liệu IMDb ban đầu
│   └── imdb_cleaned.csv             # Dữ liệu sau tiền xử lý (tạo từ Notebook)
├── Movie_Analysis.ipynb             # Phân tích EDA và huấn luyện mô hình
├── app.py                          # Ứng dụng Dashboard Streamlit
├── linear_revenue_model.pkl         # Mô hình Linear Regression đã lưu
└── README.md                        # Hướng dẫn dự án
```

## 4. Công nghệ và thư viện

- **Ngôn ngữ:** Python
- **Môi trường phân tích:** Jupyter Notebook
- **Xử lý và tính toán:** Pandas, NumPy
- **Trực quan hóa:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Lưu và tải mô hình:** Joblib
- **Ứng dụng web:** Streamlit

Cài đặt các thư viện cần thiết bằng lệnh:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit notebook
```

## 5. Hướng dẫn chạy dự án

**Bước 1 — Mở thư mục dự án:** Mở Terminal tại thư mục `Movie_Data_Analysis`, nơi chứa `app.py`.

**Bước 2 — Chạy Notebook (nếu cần tạo lại dữ liệu và mô hình):**

```bash
jupyter notebook
```

Mở `Movie_Analysis.ipynb`, chọn **Restart & Run All**. Notebook đọc dữ liệu `data/imdb_movie_dataset.csv` và tạo/cập nhật:

- `data/imdb_cleaned.csv`
- `linear_revenue_model.pkl`

**Bước 3 — Chạy Dashboard:**

```bash
streamlit run app.py
```

Mở địa chỉ do Streamlit hiển thị trên Terminal (thường là `http://localhost:8501`). Nếu đã có sẵn hai tệp `imdb_cleaned.csv` và `linear_revenue_model.pkl`, có thể chạy trực tiếp Dashboard mà không cần thực thi lại Notebook.

## 6. Mô hình dự đoán doanh thu

- **Biến mục tiêu:** `Revenue` (triệu USD).
- **Đầu vào:** `Year`, `Runtime`, `Rating`, `Votes`, `Metascore`.
- **Dữ liệu huấn luyện/kiểm tra:** 872 phim có thông tin doanh thu; chia tỷ lệ 80%/20%.
- **Mô hình triển khai:** Linear Regression.

| Chỉ số trên tập kiểm tra | Kết quả Linear Regression |
|---|---:|
| MAE | 50,68 triệu USD |
| RMSE | 79,89 triệu USD |
| R² | 0,5109 |

**Lưu ý:** 128 phim thiếu `Revenue` được giữ lại để phục vụ các phân tích không cần doanh thu; chỉ các phim có `Revenue` hợp lệ mới được dùng để huấn luyện mô hình. Giá trị thiếu của `Metascore` được thay bằng trung vị 59,5.

Mô hình sử dụng cả `Rating`, `Votes` và `Metascore` — những thông tin thường chỉ có sau khi phim phát hành. Do đó, chức năng này mang tính **ước tính doanh thu dựa trên dữ liệu IMDb**, không phải công cụ dự báo doanh thu trước khi công chiếu. Kết quả chỉ mang tính tham khảo.
