import dataclasses
import json
import os
from dataclasses import dataclass


@dataclass
class SpellCheckerConfig:
    # 1. Cấu hình đường dẫn
    model_path: str = "models"
    dict_path: str = "wordlist.dic"

    # 2. Cấu hình sinh ứng viên (Candidate Generation)
    # Quyết định có bao nhiêu từ lọt vào vòng chung kết tính điểm.
    top_n: int = 25
    # Số lượng từ ứng viên tối đa được rút ra từ từ điển cho mỗi từ gõ sai.
    # - Nếu TĂNG (VD: 50): Trễ (latency) cao hơn, chạy chậm hơn. Đổi lại, tăng cơ hội
    #   tìm thấy từ đúng ngay cả khi user gõ sai nhiều (sai cả cụm dài).
    # - Nếu GIẢM (VD: 5): Chạy cực nhanh. Nhưng nếu user gõ sai quá nhiều, từ đúng
    #   sẽ bị vứt bỏ ngay từ vòng gửi xe.

    cutoff: float = 0.4
    # Ngưỡng độ giống nhau tối thiểu về mặt chữ (Difflib Ratio) để được chọn.
    # Tính trên thang 0.0 đến 1.0 (0.4 tương đương mặt chữ giống nhau 40%).
    # - Nếu TĂNG (VD: 0.7): Chỉ cho phép các từ gõ sai rất nhẹ (sai 1-2 ký tự) đi tiếp.
    #   Bộ lọc hoạt động khắt khe, tốc độ xử lý nhanh.
    # - Nếu GIẢM (VD: 0.2): Chấp nhận cả những từ nhìn "chả liên quan gì" (sai 70-80%).
    #   Rất tốt để trị Teencode/Gõ tắt, nhưng làm tăng gánh nặng tính toán phía sau.

    # 3. Trọng số chấm điểm (Scoring Weights)
    # Cân bằng quyền lực giữa Mặt chữ và Ngữ cảnh
    sim_weight: int = 5
    # Lũy thừa áp dụng cho điểm giống nhau về mặt chữ (Similarity Score).
    # - Nếu TĂNG (VD: 10): AI trở nên rất "Bảo thủ". Nó bắt buộc từ được chọn phải
    #   có mặt chữ cực kỳ giống với từ user gõ, dominate điểm ngữ cảnh.
    # - Nếu GIẢM (VD: 1): AI dễ dãi với mặt chữ, dễ bị dominate bởi điểm ngữ cảnh.

    context_weight: float = 1.0
    # Lũy thừa áp dụng cho điểm ngữ cảnh (Bigram / Tần suất đi kèm nhau).
    # - Nếu TĂNG (VD: 2): AI trở thành "Thánh đoán ý". Nó sẵn sàng thay đổi hoàn toàn
    #   từ bạn gõ thành một từ khác hẳn (mặt chữ sai lệch nhiều) chỉ vì cụm từ đó
    #   rất phổ biến trong đời sống. (Dễ dẫn đến sửa thái quá - Overcorrection).
    # - Nếu GIẢM (VD: 0.1): AI ngây ngô hơn, từ chối sửa thành cụm có nghĩa nếu mặt chữ khác xa.

    # 4. Các hệ số nội bộ (Thuật toán Keyboard-Aware & DAMERAU-LEVENSHTEIN)
    # Mô phỏng hành vi trượt tay trên bàn phím của con người.
    max_kb_distance: float = 4.0
    # Khoảng cách vật lý tối đa trên bàn phím dùng để chuẩn hóa (chia lấy tỷ lệ).
    # - Nếu TĂNG (VD: 10.0): AI rất "khoan hồng" với lỗi gõ nhầm. Kể cả bạn gõ nhầm 'a' thành 'p'
    #   (cách nhau rất xa), nó vẫn coi là lỗi nhẹ và ít trừ điểm.
    # - Nếu GIẢM (VD: 2.0): AI rất "khắt khe". Nó chỉ tha thứ (trừ ít điểm) nếu bạn gõ nhầm
    #   các phím dính sát cạnh nhau (VD: 'a' và 's'). Còn phím cách xa nhau nó sẽ phạt thẳng tay.

    unknown_char_penalty: float = 1.0
    # Điểm phạt khi user gõ các ký tự lạ không có trên ma trận phím (VD: số 1-9, dấu @#$).
    # Phí càng cao (gần 1.0) thì AI càng ghét ký tự đó.
    # - Nếu TĂNG (VD: 2.0): AI sẽ ép bằng được các số/dấu câu về một chữ cái nào đó.
    # - Nếu GIẢM (VD: 0.1): AI sẽ có xu hướng chấp nhận giữ nguyên
    #   số hoặc ký tự lạ đó trong kết quả đầu ra.

    transposition_cost: float = 1.0  # Điểm phạt khi gõ đảo vị trí (Damerau)
    # Phí phạt khi gõ sai thứ tự 2 phím do tốc độ gõ nhanh (Lỗi Damerau).
    # VD: Gõ "thiet" thành "htiet". Phí thay thế bình thường là 2.0 (sai 2 chữ).
    # - Nếu TĂNG (VD: 1.5): AI cho rằng lộn thứ tự là một lỗi khá nặng.
    # - Nếu GIẢM (VD: 0.5): AI cực kỳ thông cảm với lỗi gõ lộn xộn do tay nhanh hơn não.
    #   Các từ bị đảo chữ sẽ giữ được điểm số cao và dễ dàng được khôi phục.

    stutter_penalty: float = 0.01
    # Hệ số nhân (phạt) khi ứng viên hiện tại giống y hệt từ đứng ngay trước nó.
    # Giúp ngăn chặn hiện tượng AI tự điền thành "trong trong", "của của".
    # - Nếu GIẢM (VD: 0.0001): Cấm tuyệt đối mô hình sinh ra 2 từ
    #   giống nhau liên tiếp. Điểm ngữ cảnh sẽ bị ép về gần 0.
    # - Nếu TĂNG (VD: 1.0): Tắt hoàn toàn hình phạt này. Mô hình có thể sinh ra 2 từ
    #   trùng nhau nếu dữ liệu train (Corpus) vô tình có những cụm từ đó.

    exact_match_bonus: float = 1.5
    # Nếu từ user gõ giống 100% với một từ trong wordlist.dic, nhân hệ số.

    auto_ambiguous_top_k: int = 20
    # Tự động loại bỏ Top K từ phổ biến nhất khỏi danh sách được phép Neo

    beam_width: int = 5  # Chỉ giữ lại x nhánh Viterbi tốt nhất mỗi bước

    # Trọng số cho N-grams
    lambda_3: float = 0.6  # Trọng số cho Trigram
    lambda_2: float = 0.3  # Trọng số cho Bigram
    lambda_1: float = 0.1  # Trọng số cho Unigram

    @classmethod
    def from_json(cls, json_path: str) -> "SpellCheckerConfig":
        """Đọc cấu hình từ file JSON (nếu có) và ghi đè lên giá trị mặc định."""
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Chỉ lấy những key có tồn tại trong định nghĩa dataclass để tránh lỗi
                valid_keys = {f.name for f in dataclasses.fields(cls)}
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}

                return cls(**filtered_data)
            except Exception as e:
                print(
                    f"Lỗi khi đọc file config {json_path}: {e}. Đang dùng config mặc định."
                )

        # Nếu file không tồn tại, trả về cấu hình mặc định
        return cls()
