import argparse
import itertools
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

from config import SpellCheckerConfig
from spellcheck import MLSpellChecker

COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

# ==============================================================================
# WORKER INITIALIZATION (Tránh load model nhiều lần gây tràn RAM)
# ==============================================================================
global_checker = None
global_test_cases = None


def create_template_files(dataset_path: str, space_path: str):
    """Tạo file mẫu trống (List/Dict) nếu chưa tồn tại"""
    if not os.path.exists(dataset_path):
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)  # Trống []
        print(f"[*] Đã tạo template data test trống tại: {dataset_path}")

    if not os.path.exists(space_path):
        with open(space_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)  # Trống {}
        print(f"[*] Đã tạo template không gian tìm kiếm trống tại: {space_path}")


def init_worker(model_path: str, test_cases: List[Dict]):
    """
    Hàm này chạy 1 lần duy nhất khi một Process (Nhân CPU) được sinh ra.
    Nó load model vào RAM riêng của Process đó.
    """
    global global_checker, global_test_cases
    # Load model vào bộ nhớ của worker
    config = SpellCheckerConfig(model_path=model_path)
    global_checker = MLSpellChecker(config=config, debug=False, detail_log=False)
    global_test_cases = test_cases


def evaluate_single_config(job_id: int, params: Dict[str, Any]) -> dict:
    """
    Hàm thực thi trên từng Process. Nhận một cấu hình và chạy trên toàn bộ test_cases.
    """
    global global_checker, global_test_cases

    # 1. Ghi đè cấu hình hiện tại vào checker của worker
    for key, value in params.items():
        if hasattr(global_checker.cfg, key):
            setattr(global_checker.cfg, key, value)

    # 2. Chạy đánh giá
    correct_count = 0
    total_cases = len(global_test_cases)

    for case in global_test_cases:
        wrong_text = case.get("wrong", "")
        expected_text = case.get("expected", "")

        # Chạy model để lấy danh sách ứng viên (List[str])
        predicted_list = global_checker.correct_sentence(wrong_text)

        # Kiểm tra xem list có rỗng không và lấy kết quả để so sánh
        if predicted_list and expected_text in predicted_list:
            correct_count += 1

    accuracy = (correct_count / total_cases) * 100

    return {
        "job_id": job_id,
        "accuracy": accuracy,
        "correct": correct_count,
        "params": params,
    }


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================


def generate_grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tune_dataset.json")
    parser.add_argument("--space", type=str, default="search_space.json")
    parser.add_argument("--model", type=str, default="models")
    parser.add_argument("--output", type=str, default="best_params.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Số nhân CPU sử dụng. Mặc định là xài toàn bộ.",
    )
    args = parser.parse_args()

    create_template_files(args.dataset, args.space)

    # Đọc dữ liệu
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        with open(args.space, "r", encoding="utf-8") as f:
            search_space = json.load(f)

        # Kiểm tra nếu file chỉ chứa mảng rỗng [] hoặc obj rỗng {}
        if not test_cases:
            print(
                f"{COLOR_YELLOW}Dataset hiện đang trống. Vui lòng nhập dữ liệu test vào '{args.dataset}' rồi chạy lại.{COLOR_RESET}"
            )
            return
        if not search_space:
            print(
                f"{COLOR_YELLOW}Search space hiện đang trống. Vui lòng nhập tham số cần tune vào '{args.space}' rồi chạy lại.{COLOR_RESET}"
            )
            return
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    param_grid = generate_grid(search_space)
    total_combinations = len(param_grid)

    print(f"\n{COLOR_CYAN}=== BẮT ĐẦU TUNING ĐA TIẾN TRÌNH ==={COLOR_RESET}")
    print(f"- Số CPU Workers : {args.workers}")
    print(f"- Cấu hình test  : {total_combinations}")
    print(f"- Câu hỏi/cấu hình: {len(test_cases)}")
    print(f"- Tổng số phép tính: {total_combinations * len(test_cases)}")

    best_accuracy = -1.0
    best_params = {}
    results_log = []

    start_time = time.time()

    # Khởi tạo Pool đa tiến trình
    # initializer sẽ chạy init_worker() trên mỗi core để load file JSON 1 lần duy nhất
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.model, test_cases),
    ) as executor:
        # Gửi toàn bộ cấu hình (job) vào cho các CPU xử lý
        future_to_job = {
            executor.submit(evaluate_single_config, i, params): i
            for i, params in enumerate(param_grid)
        }

        completed = 0
        # Nhận kết quả ngay khi có bất kỳ core nào chạy xong 1 job
        for future in as_completed(future_to_job):
            completed += 1
            result = future.result()

            results_log.append(result)

            # In tiến độ (Do chạy đa luồng nên thứ tự xong sẽ lộn xộn, ta in số lượng hoàn thành)
            print(
                f"[{completed}/{total_combinations}] Acc: {result['accuracy']:05.2f}% | Params: {result['params']}"
            )

            # Cập nhật mốc best
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_params = result["params"]

    end_time = time.time()

    print(
        f"\n{COLOR_GREEN}=== HOÀN TẤT SAU {end_time - start_time:.2f} GIÂY ==={COLOR_RESET}"
    )
    print(f"Độ chính xác cao nhất : {COLOR_GREEN}{best_accuracy:.2f}%{COLOR_RESET}")
    print(
        f"Tham số tốt nhất      : {COLOR_CYAN}{json.dumps(best_params, indent=4)}{COLOR_RESET}"
    )

    output_data = {
        "best_accuracy": best_accuracy,
        "best_params": best_params,
        "all_results": sorted(results_log, key=lambda x: x["accuracy"], reverse=True),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"[*] Đã lưu toàn bộ lịch sử chạy vào: {args.output}")


if __name__ == "__main__":
    main()
