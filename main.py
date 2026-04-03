import argparse
import sys

from config import SpellCheckerConfig
from spellcheck import MLSpellChecker
from train import load_corpus_from_folder, train_and_save_model

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def run_train(args):
    print(f"{COLOR_CYAN}BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH{COLOR_RESET}")

    full_corpus: str = load_corpus_from_folder(args.data_folder)

    if not full_corpus.strip():
        print(
            f"{COLOR_RED}Không có dữ liệu để train! Hãy thêm file .txt vào thư mục '{args.data_folder}'{COLOR_RESET}"
        )
        sys.exit(1)

    train_and_save_model(
        text_corpus=full_corpus,
        output_filename=args.model_path,
        external_dict_path=args.dict_path,
    )
    print(f"{COLOR_GREEN}Huấn luyện hoàn tất!{COLOR_RESET}")


def run_check(args):
    try:
        config = SpellCheckerConfig(model_path=args.model_path)

        checker = MLSpellChecker(
            config=config,
            debug=args.debug,
            detail_log=args.detail,
        )
    except FileNotFoundError:
        print(
            f"{COLOR_RED}Không tìm thấy file '{args.model_path}'. Hãy chạy lệnh 'python main.py train' trước!{COLOR_RESET}"
        )
        sys.exit(1)

    incorrect_sentence: str = ""
    if args.text:
        incorrect_sentence: str = args.text
    else:
        try:
            incorrect_sentence: str = input("Nhập văn bản: ")
        except KeyboardInterrupt:
            sys.exit(1)

    print(f"\n{COLOR_RED}Câu gốc: {incorrect_sentence}{COLOR_RESET}")
    corrected_sentences = checker.correct_sentence(incorrect_sentence, top_k=args.top_k)

    print(f"{COLOR_GREEN}Các gợi ý sửa lỗi:{COLOR_RESET}")
    for idx, sent in enumerate(corrected_sentences, 1):
        print(f"{COLOR_GREEN}  {idx}. {sent}{COLOR_RESET}")
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="")

    subparsers = parser.add_subparsers(
        dest="command", help="Chọn chế độ chạy (train hoặc check)", required=True
    )

    parser_train = subparsers.add_parser(
        "train", help="Huấn luyện mô hình từ thư mục data"
    )
    parser_train.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Thư mục chứa các file .txt để train (Mặc định: data)",
    )
    parser_train.add_argument(
        "--dict_path",
        type=str,
        default="wordlist.dic",
        help="Đường dẫn đến file từ điển ngoài (wordlist.dic)",
    )
    parser_train.add_argument(
        "--model_path",
        type=str,
        default="language_model.json",
        help="Tên file model xuất ra (Mặc định: language_model.json)",
    )

    parser_check = subparsers.add_parser("check", help="Chạy sửa lỗi chính tả")
    parser_check.add_argument(
        "--text",
        type=str,
        default=None,
        help="Câu cần sửa",
    )
    parser_check.add_argument(
        "--model_path",
        type=str,
        default="language_model.json",
        help="Đường dẫn đến file model",
    )
    parser_check.add_argument(
        "--dict_path",
        type=str,
        default="wordlist.dic",
        help="Đường dẫn đến file từ điển ngoài (wordlist.dic)",
    )
    parser_check.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Số lượng kết quả gợi ý trả về (Mặc định: 5)",
    )

    # Các cờ (flags) debug
    parser_check.add_argument(
        "--debug", action="store_true", help="Bật chế độ hiển thị bảng xếp hạng điểm"
    )
    parser_check.add_argument(
        "--detail",
        action="store_true",
        help="Bật chế độ hiển thị chi tiết phép tính (dùng kèm --debug)",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "check":
        run_check(args)


if __name__ == "__main__":
    main()
