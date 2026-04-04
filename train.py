import json
import os
import re
import unicodedata
from collections import Counter
from typing import List, Set

from config import ModelData

# 1. Toàn bộ nguyên âm Tiếng Việt
VOWELS = "aeiouyàáãạảăắằẵặẳâấầẫậẩèéẽẹẻêếềễệểìíĩịỉòóõọỏôốồỗộổơớờỡợởùúũụủưứừữựửỳýỹỵỷ"

# 2. Các nguyên âm mang dấu thanh (Sắc, Huyền, Hỏi, Ngã, Nặng)
# Gồm 5 dấu x 12 nguyên âm = 60 ký tự
TONED_VOWELS = "àáãạảắằẵặẳấầẫậẩèéẽẹẻếềễệểìíĩịỉòóõọỏốồỗộổớờỡợởùúũụủứừữựửỳýỹỵỷ"

# 3. Phụ âm đầu hợp lệ (Initial Consonants)
# Lưu ý: Sắp xếp các cụm dài lên trước (ngh, ch) để Regex ưu tiên match cụm dài.
INITIALS = r"(ch|gh|gi|kh|ngh|ng|nh|ph|qu|th|tr|b|c|d|đ|g|h|k|l|m|n|p|r|s|t|v|x)"

# 4. Phụ âm cuối hợp lệ (Final Consonants)
FINALS = r"(ch|ng|nh|c|m|n|p|t)"

# Biểu thức Regex kiểm tra Cấu trúc: [Phụ âm đầu] + [Nguyên âm] + [Phụ âm cuối]
# Dấu ? nghĩa là có thể không có (ví dụ: "ai" không có phụ âm đầu/cuối)
SYLLABLE_PATTERN = re.compile(rf"^{INITIALS}?([{VOWELS}]+){FINALS}?$")


def _get_toned_variations(base_char: str) -> str:
    variations = {
        "a": "àáảãạ",
        "ă": "ằắẳẵặ",
        "â": "ầấẩẫậ",
        "e": "èéẻẽẹ",
        "ê": "ềếểễệ",
        "i": "ìíỉĩị",
        "o": "òóỏõọ",
        "ô": "ồốổỗộ",
        "ơ": "ờớởỡợ",
        "u": "ùúủũụ",
        "ư": "ừứửữự",
        "y": "ỳýỷỹỵ",
    }
    return variations.get(base_char, "")


def is_valid_vietnamese_word(word: str) -> bool:
    # Đưa các ký tự NFD (tổ hợp) về dạng NFC (dựng sẵn) để Regex hoạt động chính xác.
    word = unicodedata.normalize("NFC", word)

    # Độ dài tối thiểu 1 và tối đa 7 ("nghiêng", "chương", "khuếch")
    if not (1 <= len(word) <= 7):
        return False

    # Mỗi âm tiết CHỈ CÓ TỐI ĐA 1 DẤU THANH
    # Nếu đếm ra 2 ký tự có dấu (vd: "hòá") -> Cook ngay!
    tone_count = sum(1 for char in word if char in TONED_VOWELS)
    if tone_count > 1:
        return False

    # Khớp cấu trúc (Đầu + Giữa + Cuối) & Ký tự hợp lệ
    match = SYLLABLE_PATTERN.match(word)
    if not match:
        return False

    initial = match.group(1) or ""
    vowel_part = match.group(2)

    # Cụm nguyên âm không được quá dài (Max 3 ký tự: "iêu", "oai", "uyê")
    if len(vowel_part) > 3:
        return False

    # Luật chính tả c, k / g, gh / ng, ngh
    # Nhóm nguyên âm trước (front vowels): e, ê, i, y (bao gồm cả dạng có dấu)
    front_vowel_base = "eêiy"

    # Kiểm tra xem nguyên âm đầu tiên có thuộc nhóm (e, ê, i, y) không
    first_v_char = vowel_part[0]
    # Bóc tách dấu thanh để lấy ký tự gốc so sánh
    is_front_vowel = any(
        first_v_char in _get_toned_variations(base) for base in front_vowel_base
    )

    # Luật: "gh", "ngh", "k" BẮT BUỘC phải đi với e, ê, i, y
    if initial in ["gh", "ngh", "k"] and not is_front_vowel:
        return False

    # Luật: "g", "ng", "c" KHÔNG ĐƯỢC đi với e, ê, i, y
    if initial in ["g", "ng", "c"] and is_front_vowel:
        # Ngoại lệ duy nhất: "g" đi với "i" tạo thành "gi" đã được bóc tách ở INITIALS
        # (Chữ "giêng" sẽ có initial="gi", vowel="ê", final="ng" -> Hợp lệ)
        return False

    return True


def extract_valid_sequences(raw_text: str) -> List[List[str]]:
    """
    Tách văn bản thành danh sách các chuỗi từ hợp lệ.
    Dấu câu và các "từ rác" sẽ đóng vai trò như vách ngăn, chia đứt câu.
    VD: "Hôm nay, tôi đi học." -> [["hôm", "nay"], ["tôi", "đi", "học"]]
    """
    text = raw_text.lower()

    # 1. Thay thế mọi dấu câu và ký tự ngắt dòng bằng một "vách ngăn" đặc biệt là dấu |
    # (Đảm bảo các từ sát dấu câu không bị dính vào nhau)
    text = re.sub(r'[.,!?;:()\[\]{}""\'\n\r\t\-]', " | ", text)

    raw_words = text.split()

    sequences: List[List[str]] = []
    current_seq: List[str] = []

    for w in raw_words:
        # Nếu gặp vách ngăn (dấu câu) -> Cắt đứt chuỗi hiện tại
        if w == "|":
            if current_seq:
                sequences.append(current_seq)
                current_seq = []
            continue

        # Kiểm tra từ
        if is_valid_vietnamese_word(w):
            current_seq.append(w)
        else:
            # Nếu gặp một TỪ RÁC (vd: iphone, 17), từ đó cũng sẽ trở thành vách ngăn
            # -> Cắt đứt chuỗi hiện tại để không tạo Bigram xuyên qua từ rác.
            if current_seq:
                sequences.append(current_seq)
                current_seq = []

    # Đưa chuỗi cuối cùng (nếu có) vào danh sách
    if current_seq:
        sequences.append(current_seq)

    return sequences


def train_and_save_model(
    text_corpus: str,
    output_filename="language_model.json",
    external_dict_path: str | None = None,
) -> None:
    print("Training N-grams từ Corpus...")

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    vocab_set: Set[str] = set()

    print("Đang tách câu và phân tích ngữ cảnh...")
    # Lấy ra tất cả các "cụm từ liên tiếp hợp lệ" từ Corpus
    valid_sequences = extract_valid_sequences(text_corpus)

    for seq in valid_sequences:
        len_seq: int = len(seq)

        # 1. Đếm Unigram và cập nhật Vocab
        unigram_counts.update(seq)
        vocab_set.update(seq)

        # 2. Đếm Bigram
        # Chắc chắn 100% các từ trong seq đứng sát nhau và không bị chặn bởi dấu câu nào
        if len_seq >= 2:
            bigrams = zip(seq, seq[1:])
            bigram_counts.update([f"{w1} {w2}" for w1, w2 in bigrams])

        # 3. Đếm Trigram
        if len_seq >= 3:
            trigrams = zip(seq, seq[1:], seq[2:])
            trigram_counts.update([f"{w1} {w2} {w3}" for w1, w2, w3 in trigrams])

    print(f"-> Vocab từ Corpus: {len(vocab_set)} từ.")

    if external_dict_path:
        print(f"Đang nạp từ điển ngoài: '{external_dict_path}'...")
        try:
            with open(external_dict_path, "r", encoding="utf-8") as f:
                # Chỉ lấy những từ đạt chuẩn tiếng Việt trong từ điển ngoài
                clean_external_words = []
                for line in f:
                    w = line.strip().lower()
                    if is_valid_vietnamese_word(w):
                        clean_external_words.append(w)

                vocab_set.update(clean_external_words)
            print(f"-> Đã nạp thêm từ vựng. Tổng Vocab hiện tại: {len(vocab_set)} từ.")
        except FileNotFoundError:
            print(f"File not found: '{external_dict_path}'. Skip this step.")

    # Sort theo tần suất giảm dần
    sorted_unigrams = dict(
        sorted(unigram_counts.items(), key=lambda x: x[1], reverse=True)
    )

    sorted_bigrams = dict(
        sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    )

    sorted_trigrams = dict(
        sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Vocab sort theo alphabet
    sorted_vocab = sorted(vocab_set)

    # Khởi tạo cấu trúc lưu model
    model_data: ModelData = {
        "trigrams": sorted_trigrams,
        "bigrams": sorted_bigrams,
        "unigrams": sorted_unigrams,
        "vocab": sorted_vocab,
    }

    print("Writing to JSON...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=4)

    print(f"Done! File saved to: {output_filename}")


def load_corpus_from_folder(folder_path: str) -> str:
    # Quét và đọc nội dung toàn bộ file .txt trong thư mục
    corpus_list: List[str] = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"File not found: {folder_path}")

    # Quét tất cả các file trong thư mục
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    print(f"  + Reading: {filename}...")
                    corpus_list.append(f.read())
                    file_count += 1
            except Exception as e:
                print(f"  Error while reading {filename}: {e}")

    if file_count > 0:
        print(f"Merged {file_count} file .txt!")

    # Nối tất cả nội dung lại thành 1 chuỗi khổng lồ
    return "\n".join(corpus_list)
