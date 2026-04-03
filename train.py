import json
import os
import re
from collections import Counter
from typing import List, Set, Tuple

from config import ModelData


def clean_vietnamese_text(raw_text: str) -> List[str]:
    """Lọc các từ không phải Tiếng Việt"""

    # Chuẩn hoá đầu vào
    text: str = raw_text.lower()

    # Loại bỏ các từ không có trong danh sách
    vie_chars: str = (
        "a-zàáãạảăắằẵặẳâấầẫậẩđèéẽẹẻêếềễệểìíĩịỉòóõọỏôốồỗộổơớờỡợởùúũụủưứừữựửỳýỹỵỷ"
    )
    pattern: str = rf"[^{vie_chars}\s]"
    # Các từ ví dụ như "T0i" qua sàng lọc sẽ thành ["t", "i"]
    # Các chữ cái đơn lẻ hoặc không có nghĩa khi qua bước training
    # sẽ có xác suất xuất hiện rất thấp tuỳ dữ liệu đầu vào
    text: str = re.sub(pattern, " ", text)

    # Lọc khoảng trắng
    text: str = re.sub(r"\s+", " ", text).strip()

    return text.split()


def train_and_save_model(
    text_corpus: str,
    output_filename="language_model.json",
    external_dict_path: str | None = None,
) -> None:
    print("Training N-grams từ Corpus...")

    # Lọc và trả về bộ từ khoá xếp theo thứ tự tương ứng
    # với đầu vào
    words: List[str] = clean_vietnamese_text(text_corpus)

    print("Thống kê Unigram & Bigram...")
    # Đếm số lần xuất hiện của các từ
    unigram_counts: Counter[str] = Counter(words)

    # Ghép 2 từ đứng cạnh nhau để sinh tổ hợp
    bigrams: zip[Tuple[str, str]] = zip(words, words[1:])
    # Đếm tần suất xuất hiện tổ hợp đó trong đầu vào
    bigram_counts: Counter[str] = Counter([f"{w1} {w2}" for w1, w2 in bigrams])

    vocab_set: Set[str] = set(words)

    # Duyệt theo từng dòng để không bị lọt Bigram xuyên dòng
    for line in text_corpus.split("\n"):
        words: List[str] = clean_vietnamese_text(line)
        if not words:
            continue

        unigram_counts.update(words)
        vocab_set.update(words)

        # Ghép 2 từ đứng cạnh nhau trong cùng 1 dòng
        bigrams = zip(words, words[1:])
        bigram_counts.update([f"{w1} {w2}" for w1, w2 in bigrams])

    print(f"-> Vocab từ Corpus: {len(vocab_set)} từ.")

    if external_dict_path:
        print(f"Đang nạp từ điển ngoài: '{external_dict_path}'...")
        try:
            with open(external_dict_path, "r", encoding="utf-8") as f:
                # Đọc từng dòng, xoá khoảng trắng thừa và chuyển thành chữ thường
                external_words = [line.strip().lower() for line in f if line.strip()]

                # Chạy qua hàm clean để đảm bảo từ điển ngoài không dính rác
                clean_external_words = []
                for w in external_words:
                    clean_external_words.extend(clean_vietnamese_text(w))

                # Gộp vào tập Vocab hiện tại
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

    # Vocab sort theo alphabet
    sorted_vocab = sorted(vocab_set)

    # Khởi tạo cấu trúc lưu model
    model_data: ModelData = {
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
