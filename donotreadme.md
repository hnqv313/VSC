# 🚀 Dự Án Máy Học: Hệ Thống Tự Động Sửa Lỗi Chính Tả Tiếng Việt (Vietnamese Spell Checker)

Một công cụ máy học (Machine Learning) mạnh mẽ, tốc độ cao giúp phát hiện và tự động sửa lỗi chính tả cho văn bản Tiếng Việt. Dự án được thiết kế đặc biệt để xử lý các đặc thù của tiếng Việt, bao gồm lỗi gõ Telex, lỗi nhầm lẫn vị trí phím vật lý trên bàn phím QWERTY, và sửa lỗi dựa trên ngữ cảnh (context-aware) bằng Mô hình Ngôn ngữ (Language Model).

---

## 1. 🎯 Problem Definition (Định nghĩa bài toán)

**Vấn đề:** Người dùng khi nhập liệu tiếng Việt thường xuyên mắc phải các loại lỗi chính tả sau:

1. **Lỗi do cách gõ (Typo):** Gõ trượt tay sang các phím lân cận trên bàn phím QWERTY (ví dụ: `s` thay vì `a`, `j` thay vì `h`).
2. **Lỗi quy tắc gõ tiếng Việt (Telex/VNI):** Gõ sai thứ tự dấu hoặc sai quy tắc tổ hợp chữ (ví dụ: thay vì gõ "người" lại gõ sai thành "nguwowfi", "đẹp" thành "đpẹ").
3. **Lỗi ngữ cảnh:** Chữ gõ ra có ý nghĩa nhưng không hợp lý trong ngữ cảnh của câu (ví dụ: "con *sâu* gặm lá" vs "con *sâu* xa").

**Mục tiêu:** Xây dựng một hệ thống có khả năng nhận đầu vào là một câu bị lỗi chính tả và tự động sinh ra (predict) câu đúng nghĩa nhất, cấu trúc chuẩn xác nhất trong thời gian thực (real-time) với tài nguyên tính toán tối thiểu.

---

## 2. 🧠 Method Used (Phương pháp và Cách hoạt động)

Dự án áp dụng phương pháp kết hợp giữa **Xử lý ngôn ngữ tự nhiên (NLP) thống kê**, **Fuzzy Matching**, và **Tìm kiếm Heuristic**.

### Cách hoạt động cốt lõi

1. **Sinh ứng viên (Candidate Generation):** - Với mỗi từ gõ sai, hệ thống sử dụng thuật toán tính khoảng cách chuỗi (qua thư viện `rapidfuzz`) để quét trong từ điển (`wordlist.dic`) và trích xuất ra Top N ứng viên giống về mặt chữ nhất.
   - Điểm đặc biệt: Ứng dụng quy chuẩn Telex (`to_standard_telex` trong `text_utils.py`) để đối chiếu, giúp mô hình hiểu được ý định gõ dấu của người dùng.
2. **Tính điểm lỗi vật lý (Keyboard Layout Distance):**
   - Áp dụng ma trận tọa độ bàn phím QWERTY (`layout.py`) để tính toán xem việc gõ nhầm chữ A thành chữ S có xác suất cao hơn hay không (do 2 phím này nằm cạnh nhau).
3. **Mô hình ngôn ngữ N-gram (N-gram Language Model):**
   - Chấm điểm ngữ cảnh dựa trên xác suất xuất hiện của chuỗi từ thông qua mô hình Unigram, Bigram, và Trigram. Điểm số được tính bằng phương pháp nội suy tuyến tính (Linear Interpolation) với các trọng số $\lambda_1, \lambda_2, \lambda_3$ cấu hình linh hoạt.
4. **Giải mã (Decoding) bằng Beam Search / Viterbi:**
   - Thay vì chọn từ đúng nhất cục bộ, hệ thống dùng thuật toán Beam Search (`beam_width` = 5) để duy trì các tuyến đường (paths) tốt nhất cho cả câu, từ đó tìm ra kết hợp từ tối ưu nhất về mặt ngữ cảnh tổng thể.

---

## 3. ⚡ Đặc điểm nổi bật và Tối ưu hóa (Features & Optimizations)

- **Tối ưu hóa Bộ nhớ với Trie:** Sử dụng `marisa-trie` thay vì Python Dictionary thông thường để lưu trữ hàng triệu N-grams. Các mô hình (`unigrams.trie`, `bigrams.trie`,...) được tải dạng Memory-mapped (mmap) giúp tiết kiệm RAM tuyệt đối và load model cực nhanh.
- **Cơ chế Caching thông minh:** Các hàm tính toán nặng hoặc lặp lại nhiều lần (như chuyển đổi Telex `to_standard_telex`) được bọc bởi `@lru_cache` để tối đa hóa tốc độ phản hồi.
- **Cấu hình siêu linh hoạt (`config.py`):** Dễ dàng điều chỉnh độ khắt khe của bộ lọc (`cutoff`), số lượng ứng viên (`top_n`), trọng số n-gram, hoặc tinh chỉnh khoảng thưởng exact match theo dạng `[min_bonus, max_bonus]`.
- **Đa luồng (Multiprocessing):** Quá trình tìm kiếm siêu tham số (Tuning) chạy song song trên nhiều CPU cores thông qua `ProcessPoolExecutor`, rút ngắn thời gian tinh chỉnh mô hình.

---

## 4. 🗂️ Dataset (Dữ liệu huấn luyện)

Hệ thống không cần các bộ dữ liệu gán nhãn khổng lồ phức tạp, mà chỉ cần:

1. **Text Corpus (Tập văn bản thô):** Các file `.txt` chứa văn bản tiếng Việt đúng chuẩn được đặt trong thư mục `data/`. Module `train.py` sẽ quét, làm sạch (lọc nguyên âm, phụ âm hợp lệ bằng Regex), đếm tần suất và xây dựng N-gram model tự động.
2. **External Dictionary (Từ điển ngoài):** Một file `wordlist.dic` dùng làm bộ từ vựng mỏ neo để hệ thống sinh ứng viên khi phát hiện ra từ lạ.

---

## 5. 📊 Model Evaluation (Đánh giá mô hình)

Dự án tích hợp sẵn module `tune.py` đóng vai trò là hệ thống đánh giá và tinh chỉnh tự động (AutoML/Hyperparameter Tuning):
- **Đánh giá tự động:** Chạy tập test cases chứa các câu gõ sai phổ biến (Input) và câu đúng kỳ vọng (Expected Output).
- **Grid Search / Random Search:** Thử nghiệm kết hợp hàng ngàn bộ tham số khác nhau (ví dụ: thay đổi $\lambda_1, \lambda_2, \lambda_3$, ngưỡng `cutoff`, `beam_width`).
- **Metric (Độ đo):** Tính toán độ chính xác tổng thể (**Accuracy %**) — tỷ lệ số câu được mô hình sửa khớp hoàn toàn với nhãn gốc.
- Kịch bản sẽ tự động lưu lại tham số cho kết quả cao nhất (`best_params`) vào file log.

---

## 6. 💻 Demo (Ví dụ sử dụng)

Hệ thống cung cấp giao diện dòng lệnh (CLI) thân thiện qua file `main.py`.

**1. Huấn luyện mô hình từ dữ liệu thô:**

```bash
python main.py train --data_folder ./my_corpus_folder --model_path ./models
```

**2. Tinh chỉnh (Tuning) siêu tham số:**

```bash
python main.py tune --dataset data/test_cases.json --space space.json --workers 4
```

**3. Chạy Demo sửa lỗi chính tả:**

```bash
python main.py check --text "hmo nay tời đpẹ qáu, chugn ta đi chwoi nhe" --debug
```

**Output dự kiến (có bật cờ `--debug` để xem giải thuật):**

```text
----------------------------------------------------------------------------------
| ỨNG VIÊN     | TUYẾN TỐT NHẤT       | LỊCH SỬ + ĐIỂM BƯỚC NHẢY  | TỔNG ĐIỂM    |
----------------------------------------------------------------------------------
| hôm          | hôm                  | <Bắt đầu>                 | -4.2140      |
| nay          | hôm nay              | hôm -> nay                | -5.1123      |
| trời         | hôm nay trời         | nay -> trời               | -5.8912      |
| đẹp          | hôm nay trời đẹp     | trời -> đẹp               | -6.3210      |
| quá          | hôm nay trời đẹp quá | đẹp -> quá                | -7.1021      |
...
----------------------------------------------------------------------------------
=> KẾT QUẢ CUỐI CÙNG: "hôm nay trời đẹp quá, chúng ta đi chơi nhé"
```
