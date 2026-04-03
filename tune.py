from config import SpellCheckerConfig
from spellcheck import MLSpellChecker


def run_evaluation(checker_name: str, checker: MLSpellChecker, test_cases: list):
    print(f"\n{'=' * 50}")
    print(f"ĐANG CHẠY CẤU HÌNH: {checker_name}")
    print(f"{'=' * 50}")

    correct_count = 0

    for wrong_text, expected_text in test_cases:
        # Chạy model để sửa câu
        predicted_text = checker.correct_sentence(wrong_text)

        # Đánh giá đúng/sai
        is_correct = predicted_text == expected_text
        if is_correct:
            correct_count += 1
            status = "✅ ĐÚNG"
        else:
            status = "❌ SAI"

        # In kết quả
        print(f"Câu gốc   : {wrong_text}")
        print(f"Bot sửa   : {predicted_text} ({status})")
        print(f"Kỳ vọng   : {expected_text}")
        print("-" * 40)

    accuracy = (correct_count / len(test_cases)) * 100
    print(f"Độ chính xác tổng: {accuracy:.1f}% ({correct_count}/{len(test_cases)} câu)")


def main():
    # Chuẩn bị tập dữ liệu nhỏ để test (Ground Truth)
    # Cố tình chọn những câu gõ sai mặt chữ khá nặng (cần ngữ cảnh để cứu)
    test_cases = [
        ("nhasf pshdn tishuc cuar coong ty", "nhà phân tích của công ty"),
        ("giar xnawfg hoom nay tanng manh", "giá xăng hôm nay tăng mạnh"),
        ("tongr thoong myx phats bieu", "tổng thống mỹ phát biểu"),
    ]

    # BƯỚC 1: CHẠY THỬ VỚI CẤU HÌNH MẶC ĐỊNH
    print("Đang load model mặc định...")
    default_config = SpellCheckerConfig()

    checker_default = MLSpellChecker(config=default_config)
    run_evaluation("MẶC ĐỊNH", checker_default, test_cases)

    # BƯỚC 2: OVERRIDE CẤU HÌNH & CHẠY LẠI
    # Giả sử mô hình mặc định quá bảo thủ (chỉ tin mặt chữ),
    # Ta ghi đè cấu hình: Tin tưởng ngữ cảnh hơn (context_weight = 4)
    # và bớt soi xét mặt chữ (sim_weight = 1).
    print("\nĐang load model tinh chỉnh...")
    tuned_config = SpellCheckerConfig(
        sim_weight=1,  # Giảm trọng số mặt chữ
        context_weight=4.0,  # Tăng trọng số ngữ cảnh
        cutoff=0.3,  # Hạ ngưỡng lọc để chấp nhận từ sai lệch nhiều hơn
    )

    checker_tuned = MLSpellChecker(config=tuned_config)
    run_evaluation("TINH CHỈNH", checker_tuned, test_cases)

    # BƯỚC 3: NẾU CÁC THAO TÁC TINH CHỈNH KHÔNG HOẠT ĐỘNG
    # BẬT DEBUG MODE VÀ DETAIL TRONG MLSpellChecker NẾU CẦN, THÊM DATA


if __name__ == "__main__":
    main()
