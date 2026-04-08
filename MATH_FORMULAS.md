# Tổng hợp công thức toán học trong dự án VSC

Tài liệu này tổng hợp các công thức và biểu thức toán học đang được sử dụng trực tiếp trong mã nguồn.

## 1. Khoảng cách Euclid trên bàn phím

Nguồn: [spellcheck.py](spellcheck.py#L200), [layout.py](layout.py#L20)

Khi hai ký tự `char1`, `char2` có tọa độ bàn phím lần lượt là `(x_1, y_1)` và `(x_2, y_2)`, khoảng cách được tính là:

```text
dist = sqrt((x1 - x2)^2 + (y1 - y2)^2)
```

Sau đó chuẩn hóa về đoạn `[0, 1]`:

```text
kb_cost = min(dist / max_kb_distance, 1.0)
```

Ý nghĩa:
- Phím càng gần nhau thì chi phí thay thế càng thấp.
- Nếu ký tự không có trong ma trận bàn phím thì dùng `unknown_char_penalty`.

## 2. Damerau-Levenshtein có trọng số bàn phím

Nguồn: [spellcheck.py](spellcheck.py#L216)

Ma trận quy hoạch động `dp[i][j]` lưu chi phí nhỏ nhất để biến đổi `word1[:i]` thành `word2[:j]`:

```text
dp[i][j] = min(
  dp[i-1][j] + 1,
  dp[i][j-1] + 1,
  dp[i-1][j-1] + kb_cost(word1[i-1], word2[j-1])
)
```

Nếu có đảo chỗ hai ký tự liên tiếp:

```text
dp[i][j] = min(dp[i][j], dp[i-2][j-2] + transposition_cost)
```

Ý nghĩa:
- Xóa và chèn có chi phí cố định `1`.
- Thay thế dùng chi phí phụ thuộc vị trí phím.
- Đảo vị trí hai ký tự dùng chi phí riêng `transposition_cost`.

## 3. Chuyển khoảng cách thành độ tương đồng

Nguồn: [spellcheck.py](spellcheck.py#L250)

Sau khi có khoảng cách chỉnh sửa `distance = dp[m][n]`, độ tương đồng được đổi bằng hàm mũ:

```text
sim = exp(-distance / max(m, n))
```

Trong đó `m = len(word1)`, `n = len(word2)`.

Ý nghĩa:
- Nếu hai từ giống hệt nhau thì `distance = 0`, suy ra `sim = 1`.
- Khoảng cách càng lớn thì độ tương đồng giảm theo hàm mũ.

## 4. Xác suất ngữ cảnh N-gram

Nguồn: [spellcheck.py](spellcheck.py#L261)

### Trigram

```text
P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)
```

### Bigram

```text
P(w3 | w2) = count(w2, w3) / count(w2)
```

### Unigram

```text
P(w3) = count(w3) / total_unigrams
```

Ý nghĩa:
- Trigram đo xác suất từ kế tiếp theo hai từ trước đó.
- Bigram là phương án lùi khi ngữ cảnh ngắn hơn.
- Unigram biểu diễn mức phổ biến tổng quát của một từ.

## 5. Nội suy xác suất ngữ cảnh

Nguồn: [spellcheck.py](spellcheck.py#L291), [config.py](config.py#L88)

Khi đủ ngữ cảnh trigram:

```text
p_ctx = lambda_3 * p_tri + lambda_2 * p_bi + lambda_1 * p_uni
```

Khi chưa có `w1`:

```text
p_ctx = (lambda_2 * p_bi + lambda_1 * p_uni) / (lambda_2 + lambda_1)
```

Ý nghĩa:
- Kết hợp đồng thời tín hiệu từ trigram, bigram và unigram.
- Bộ trọng số mặc định là `lambda_3 = 0.6`, `lambda_2 = 0.3`, `lambda_1 = 0.1`.

## 6. Phạt lặp từ liên tiếp

Nguồn: [spellcheck.py](spellcheck.py#L330), [config.py](config.py#L71)

Nếu ứng viên hiện tại trùng từ đứng trước:

```text
p_ctx = p_ctx * stutter_penalty
```

Ý nghĩa:
- Giảm mạnh xác suất ngữ cảnh để tránh kết quả kiểu `"trong trong"`.

## 7. Biến đổi đặc trưng trước khi chấm điểm

Nguồn: [spellcheck.py](spellcheck.py#L320)

Đặc trưng mặt chữ:

```text
sim_feat = log(sim + 1e-8)
```

Đặc trưng ngữ cảnh:

```text
ctx_feat = log(p_ctx + 1e-8)
ctx_feat = (ctx_feat + 10) / 10
```

Ý nghĩa:
- Dùng log để nén miền giá trị xác suất/độ tương đồng.
- `1e-8` là hằng số tránh `log(0)`.
- `ctx_feat` được tịnh tiến và co giãn thêm để dễ phối hợp với trọng số.

## 8. Hàm điểm cuối cùng

Nguồn: [spellcheck.py](spellcheck.py#L344), [config.py](config.py#L31)

Điểm của một ứng viên:

```text
score = sim_weight * sim_feat + context_weight * ctx_feat
```

Nếu ứng viên trùng chính xác với từ gốc và có trong từ điển chuẩn:

```text
score = score + exact_match_bonus
```

Ý nghĩa:
- Đây là hàm chấm điểm log-linear kết hợp mặt chữ và ngữ cảnh.
- `sim_weight` điều khiển mức ưu tiên giống mặt chữ.
- `context_weight` điều khiển mức ưu tiên ngữ cảnh.

## 9. Tích lũy điểm trong Viterbi / Beam Search

Nguồn: [spellcheck.py](spellcheck.py#L454)

Điểm của một đường đi mới được cộng dồn theo lịch sử:

```text
total_score = prev_score + step_score
```

Ý nghĩa:
- Mỗi câu ứng viên được xem như một chuỗi quyết định.
- Thuật toán giữ lại các đường đi có tổng điểm cao nhất theo `beam_width`.

## 10. Điểm đánh giá trong quá trình tuning

Nguồn: [tune.py](tune.py#L61)

Nếu đáp án đúng xuất hiện ở vị trí `i` trong danh sách dự đoán:

```text
case_score = 1 / (i + 1)
```

Tổng độ chính xác:

```text
accuracy = (total_score / total_cases) * 100
```

Ý nghĩa:
- Đúng ở vị trí đầu tiên được `1.0` điểm.
- Đúng ở vị trí thấp hơn vẫn được thưởng nhưng ít hơn.
- `accuracy` ở đây thực chất là điểm top-k có trọng số theo thứ hạng, không phải accuracy nhị phân thuần túy.

## 11. Các ràng buộc hình thức có tính "công thức"

Nguồn: [train.py](train.py#L24)

### Mẫu âm tiết tiếng Việt

```text
SYLLABLE_PATTERN = ^INITIALS?([VOWELS]+)FINALS?$
```

Có thể hiểu khái quát là:

```text
Âm tiết = [phụ âm đầu tùy chọn] + [cụm nguyên âm] + [phụ âm cuối tùy chọn]
```

### Ràng buộc độ dài và số dấu

Nguồn: [train.py](train.py#L51)

```text
1 <= len(word) <= 7
tone_count <= 1
len(vowel_part) <= 3
```

Ý nghĩa:
- Đây không phải công thức tính điểm, nhưng là các điều kiện toán học/ràng buộc rõ ràng đang được dùng để xác thực từ tiếng Việt.

## Ghi chú

- Bộ lọc ứng viên dùng `RapidFuzz` với ngưỡng `cutoff`, nhưng công thức similarity cụ thể của thư viện không được định nghĩa trong repo nên không được liệt kê chi tiết ở đây.
