import argparse
import os
import sys

from spell_checker import NGramSpellChecker
from spell_checker_config import SpellCheckerConfig
from realtime_input import run_realtime_input

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def run_build(args):
    from language_model_builder import build_language_stats_from_folder

    print(f"{COLOR_CYAN}BẮT ĐẦU XÂY DỰNG THỐNG KÊ NGÔN NGỮ{COLOR_RESET}")
    try:
        build_language_stats_from_folder(
            folder_path=args.data_folder,
            output_dir=args.stats_path,
            external_dict_path=args.dict_path,
            num_workers=args.workers,
        )
    except ValueError:
        print(
            f"{COLOR_RED}Không có dữ liệu để xử lý. Hãy thêm file .txt vào thư mục '{args.data_folder}'{COLOR_RESET}"
        )
        sys.exit(1)
    print(f"{COLOR_GREEN}Xây dựng thống kê hoàn tất!{COLOR_RESET}")


def run_check(args):
    try:
        config = SpellCheckerConfig.from_json(args.config)

        if args.stats_path:
            config.stats_path = args.stats_path
        if args.dict_path:
            config.dict_path = args.dict_path

        if not os.path.exists(config.stats_path):
            print(
                f"{COLOR_RED}Lỗi: Không tìm thấy thư mục dữ liệu thống kê tại '{config.stats_path}'.{COLOR_RESET}"
            )
            print(
                f"{COLOR_YELLOW}Hãy chạy lệnh 'build' trước hoặc kiểm tra lại đường dẫn.{COLOR_RESET}"
            )
            return

        checker = NGramSpellChecker(
            config=config,
            debug=args.debug,
            detail_log=args.detail,
        )

        if args.realtime:
            incorrect_sentence = run_realtime_input(checker)
            return

        while True:
            try:
                incorrect_sentence = args.text or input("Nhập văn bản: ")
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
                        print(f"{COLOR_GREEN}{cw}{COLOR_RESET}", end="")
                    else:
                        print(f"{COLOR_YELLOW}{cw}{COLOR_RESET}", end="")

                    if i < len(corr_words) - 1:
                        print(" ", end="")
                print()
            print("\n")

            if args.text:
                break
    except Exception as e:
        print(f"{COLOR_RED}Lỗi vận hành: {e}{COLOR_RESET}")


def main():
    parser = argparse.ArgumentParser(description="")

    subparsers = parser.add_subparsers(
        dest="command", help="Chọn chế độ chạy (build hoặc check)", required=True
    )

    parser_build = subparsers.add_parser(
        "build", help="Xây dựng thống kê ngôn ngữ từ thư mục data"
    )
    parser_build.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Thư mục chứa các file .txt nguồn (Mặc định: data)",
    )
    parser_build.add_argument(
        "--dict_path",
        type=str,
        default="wordlist.dic",
        help="Đường dẫn đến file từ điển ngoài (wordlist.dic)",
    )
    parser_build.add_argument(
        "--stats_path",
        type=str,
        default="language_stats",
        help="Thư mục xuất dữ liệu thống kê (Mặc định: language_stats)",
    )
    parser_build.add_argument(
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
        "--realtime",
        action="store_true",
        help="Bật chế độ nhập realtime, tách ngữ cảnh theo dấu câu và tô màu từ đã sửa",
    )
    parser_check.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Đường dẫn đến file cấu hình JSON (Mặc định: config.json)",
    )
    parser_check.add_argument(
        "--stats_path",
        type=str,
        default="language_stats",
        help="Đường dẫn đến thư mục dữ liệu thống kê",
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

    parser_check.add_argument(
        "--debug", action="store_true", help="Bật chế độ hiển thị bảng xếp hạng điểm"
    )
    parser_check.add_argument(
        "--detail",
        action="store_true",
        help="Bật chế độ hiển thị chi tiết phép tính (dùng kèm --debug)",
    )

    args = parser.parse_args()

    if args.command == "build":
        run_build(args)
    elif args.command == "check":
        run_check(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
