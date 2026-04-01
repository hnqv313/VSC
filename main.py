import argparse
import sys
from typing import List

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
        checker = MLSpellChecker(
            model_path=args.model_path,
            top_n=args.top_n,
            cutoff=args.cutoff,
            sim_weight=args.sim_weight,
            context_weight=args.context_weight,
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
    corrected_sentence = checker.correct_sentence(incorrect_sentence)
    print(f"{COLOR_GREEN}Câu sửa: {corrected_sentence}{COLOR_RESET}\n")


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
        default=None,
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

    # Các siêu tham số (Hyperparameters)
    parser_check.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Số lượng ứng viên sinh ra tối đa (Mặc định: 20)",
    )
    parser_check.add_argument(
        "--cutoff", type=float, default=0.4, help="Ngưỡng giống mặt chữ (Mặc định: 0.4)"
    )
    parser_check.add_argument(
        "--sim_weight",
        type=int,
        default=3,
        help="Trọng số phạt mặt chữ sai (Mặc định: mũ 3)",
    )
    parser_check.add_argument(
        "--context_weight",
        type=int,
        default=2,
        help="Trọng số thưởng ngữ cảnh đúng (Mặc định: mũ 2)",
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
