import argparse
import os
import sys

from config import SpellCheckerConfig
from spellcheck import MLSpellChecker
from train import train_from_folder

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def run_train(args):
    print(f"{COLOR_CYAN}BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH{COLOR_RESET}")
    try:
        train_from_folder(
            folder_path=args.data_folder,
            output_dir=args.model_path,
            external_dict_path=args.dict_path,
            num_workers=args.workers,
        )
    except ValueError:
        print(
            f"{COLOR_RED}Không có dữ liệu để train! Hãy thêm file .txt vào thư mục '{args.data_folder}'{COLOR_RESET}"
        )
        sys.exit(1)
    print(f"{COLOR_GREEN}Huấn luyện hoàn tất!{COLOR_RESET}")


def run_check(args):
    try:
        config = SpellCheckerConfig.from_json(args.config)

        # Ghi đè bằng tham số dòng lệnh (nếu người dùng có nhập)
        if args.model_path:
            config.model_path = args.model_path
        if args.dict_path:
            config.dict_path = args.dict_path

        # Kiểm tra sự tồn tại của file model trước khi khởi tạo
        if not os.path.exists(config.model_path):
            print(
                f"{COLOR_RED}Lỗi: Không tìm thấy file model tại '{config.model_path}'.{COLOR_RESET}"
            )
            print(
                f"{COLOR_YELLOW}Hãy chạy lệnh 'train' trước hoặc kiểm tra lại đường dẫn.{COLOR_RESET}"
            )
            return

        checker = MLSpellChecker(
            config=config,
            debug=args.debug,
            detail_log=args.detail,
        )

        while True:
            try:
                incorrect_sentence: str = args.text or input("Nhập văn bản: ")
            except KeyboardInterrupt:
                sys.exit(1)

            print(f"\n{COLOR_RED}Câu gốc: {incorrect_sentence}{COLOR_RESET}")
            corrected_sentences = checker.correct_sentence(
                incorrect_sentence, top_k=args.top_k
            )

            print(f"{COLOR_GREEN}Các gợi ý sửa lỗi:{COLOR_RESET}")
            orig_words = incorrect_sentence.split()
            for idx, sent in enumerate(corrected_sentences, 1):
                corr_words = sent.split()
                print(f"  {idx}. ", end="")
                for i, (ow, cw) in enumerate(zip(orig_words, corr_words)):
                    if ow.lower() == cw.lower():
                        # Giữ nguyên
                        print(f"{COLOR_GREEN}{cw}{COLOR_RESET}", end="")
                    else:
                        # In màu vàng cho từ được sửa để dễ nhận biết
                        print(f"{COLOR_YELLOW}{cw}{COLOR_RESET}", end="")

                    if i < len(corr_words) - 1:
                        print(" ", end="")
                print()
            print("\n")

            if args.text:
                break
    except Exception as e:
        # raise e
        print(f"{COLOR_RED}Lỗi vận hành: {e}{COLOR_RESET}")


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
        default="models",
        help="Tên file model xuất ra (Mặc định: models)",
    )
    parser_train.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Số tiến trình xử lý song song theo file .txt (Mặc định: 1)",
    )

    parser_check = subparsers.add_parser("check", help="Chạy sửa lỗi chính tả")
    parser_check.add_argument(
        "--text",
        type=str,
        default=None,
        help="Câu cần sửa",
    )
    parser_check.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Đường dẫn đến file cấu hình JSON (Mặc định: config.json)",
    )
    parser_check.add_argument(
        "--model_path",
        type=str,
        default="models",
        help="Đường dẫn đến thư mục model",
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
        help="Số lượng kết quả gợi ý trả về trong giới hạn beam width (Mặc định: 5)",
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
    try:
        main()
    except KeyboardInterrupt:
        pass
