# Bài Toán Luồng Cực Đại trong Mạng Hậu Cần

## 1. Giới Thiệu Bài Toán

### Bối Cảnh
Một trung tâm điều khiển giao thông thông minh cần tối ưu hóa luồng xe từ khu vực nguồn (S) đến khu vực đích (T) thông qua một mạng lưới các nút giao thông và tuyến đường kết nối.

### Cấu Trúc Mạng
- **Điểm Nguồn (S)**: Khu vực phát sinh giao thông (khu dân cư/công nghiệp), có lưu lượng 120 xe/giờ
- **Các Nút Giao Thông**:
  - Nút A: Nút giao lớn kết nối nhiều hướng
  - Nút B: Trung tâm phân luồng
  - Nút C: Nút giao cấp phường/xã
  - Nút D: Nút giao chính
  - Nút E: Nút giao song song
- **Điểm Đích (T)**: Khu vực đích (trung tâm thành phố)

### Công Suất Các Tuyến (gói hàng/ngày)

- S → A: 50 (đại lộ chính)
- A → D: 40 (đường nội đô)
- C → E: 20 (đường nhánh)
- A → B: 20 (đường kết nối)
- S → B: 40 (đường vành đai)
- B → D: 50 (trục chính)
- E → T: 40 (đường vào trung tâm)
- D → E: 30 (đường một chiều)
- S → C: 30 (đường khu vực)
- C → D: 20 (đường nội bộ)
- B → E: 30 (đường liên khu vực)
- D → T: 60 (đại lộ trung tâm)

### Mạng Giao Thông
![Mạng Giao Thông](./images/network.png)

### Mục Tiêu
Tìm phương án phân luồng giao thông tối ưu để:
- Tối đa hóa lưu lượng xe có thể di chuyển từ khu vực nguồn đến đích

- Không vượt quá khả năng thông qua của mỗi tuyến đường

- Cân bằng tải trọng giao thông trên toàn mạng

## 2. Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8 trở lên
- Git

### Các Bước Cài Đặt

1. **Clone Repository**

```bash
git clone [URL_repository]
cd [tên_thư_mục]
```

2. **Tạo Môi Trường Ảo**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Cài Đặt Thư Viện**
```bash
pip install networkx matplotlib numpy tkinter ipython
```

4. **Khởi Chạy Chương Trình**
```bash
python main.py
```

### Sử Dụng Chương Trình

![Giao Diện Chính](./images/main.png)

1. **Giao Diện Chính**
- Bên trái: Hiển thị đồ thị mạng
- Bên phải: Bảng điều khiển và thống kê

2. **Các Tham Số Điều Chỉnh**
- Population Size: Số lượng phương án khả thi
- Mutation Rate: Tỷ lệ thay đổi ngẫu nhiên
- Balancing Factor: Hệ số cân bằng tải
- Truncation Rate: Tỷ lệ loại bỏ phương án kém
- Max Generations: Số lần tối ưu tối đa
- Display Delay: Độ trễ hiển thị (ms)

3. **Điều Khiển**
- Start: Bắt đầu mô phỏng
- Stop: Dừng mô phỏng
- Apply: Áp dụng thay đổi tham số

4. **Thông Tin Hiển Thị**
- Generation: Lần tối ưu hiện tại

- Current Flow: Lưu lượng hiện tại

- Best Flow: Lưu lượng tốt nhất tìm được

### Lưu Ý
- Đảm bảo tất cả thư viện được cài đặt đầy đủ
- Có thể điều chỉnh các tham số trong quá trình chạy
- Chương trình sẽ hiển thị kết quả tốt nhất sau khi hoàn thành


## Tham khảo
- [A Genetic Algorithm Applied to the Maximum Flow Problem](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=64c72556c1ed39534ed085c117cb881162ab03d0)
- [Codeforce Max Flow tutorial](https://codeforces.com/blog/entry/105330)
- [Max Flow Min Cut Theorem](https://www.geeksforgeeks.org/max-flow-problem-introduction/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [Networkx](https://networkx.org/documentation/stable/reference/classes/all.html)
- [Linear Programming: Foundations and Extensions](https://www.amazon.com/Linear-Programming-Foundations-Extensions-David-Goldberg/dp/0471383497)
## Documentation
- [Documentation](https://docs.google.com/document/d/1GS3FqorM6rPBtcU-r2f7IeRFNeqTVMBy_1WWFJNYoR0/edit?tab=t.0#heading=h.kxamao79tale)

## Thành viên đóng góp
- Huỳnh Thiên Văn
- Hồ Ngọc Bảo
- Tô Xuân Đông
- Lương Quốc Trung