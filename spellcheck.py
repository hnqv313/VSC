import json
import math
import os
import unicodedata
from typing import Dict, List, Set, Tuple

import Levenshtein
import marisa_trie

from config import SpellCheckerConfig
from layout import get_keyboard_coordinates, keyboard_matrix
from text_utils import to_standard_telex


class MLSpellChecker:
    def __init__(
        self,
        config: SpellCheckerConfig,
        debug: bool = False,
        detail_log: bool = False,
    ) -> None:
        self.cfg = config
        self.debug = debug
        self.detail_log = detail_log

        print(f"Loading model từ thư mục: {self.cfg.model_path}...")

        # Load Metadata
        meta_path = os.path.join(self.cfg.model_path, "model_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.vocab: Set[str] = set(
            unicodedata.normalize("NFC", w) for w in meta["vocab"]
        )
        self.total_unigrams = meta["total_unigrams"]

        self.unigrams = marisa_trie.RecordTrie("<I").mmap(
            os.path.join(self.cfg.model_path, "unigrams.trie")
        )
        self.bigrams = marisa_trie.RecordTrie("<I").mmap(
            os.path.join(self.cfg.model_path, "bigrams.trie")
        )

        tri_path = os.path.join(self.cfg.model_path, "trigrams.trie")
        if os.path.exists(tri_path):
            self.trigrams = marisa_trie.RecordTrie("<I").mmap(tri_path)
        else:
            self.trigrams = None

        self.telex_to_vocab: Dict[str, List[str]] = {}
        for w in self.vocab:
            t = to_standard_telex(w)
            if t not in self.telex_to_vocab:
                self.telex_to_vocab[t] = []
            self.telex_to_vocab[t].append(w)

        self.telex_vocab_list: List[str] = list(self.telex_to_vocab.keys())

        self.standard_dict: Set[str] = set()
        if self.cfg.dict_path:
            try:
                with open(self.cfg.dict_path, "r", encoding="utf-8") as f:
                    self.standard_dict = {
                        line.strip().lower() for line in f if line.strip()
                    }
                print(
                    f"Đã nạp {len(self.standard_dict)} từ chuẩn từ '{self.cfg.dict_path}'."
                )
            except FileNotFoundError:
                print(
                    f"Không tìm thấy từ điển '{self.cfg.dict_path}'. Tính năng liên quan sẽ bị vô hiệu hoá."
                )

        sorted_unigrams = sorted(
            self.unigrams.items(), key=lambda item: item[1], reverse=True
        )
        top_k = getattr(self.cfg, "auto_ambiguous_top_k", 50)
        self.dynamic_ambiguous_words: Set[str] = {
            word for word, _ in sorted_unigrams[:top_k]
        }
        if self.debug:
            print(
                f"Tự động cấm Anchor {len(self.dynamic_ambiguous_words)} từ phổ biến: {list(self.dynamic_ambiguous_words)[:10]}..."
            )

        self.kb_coords = get_keyboard_coordinates(keyboard_matrix=keyboard_matrix)
        print(f"Done! The dictionary has {len(self.vocab)} words.")

        # Tìm Log của từ xuất hiện nhiều nhất để chuẩn hóa Tần suất về 0.0 -> 1.0
        max_count = 1
        for _, v in self.unigrams.items():
            if v[0] > max_count:
                max_count = v[0]
        self.max_unigram_log = math.log2(max_count + 1)

        self._sim_cache: Dict[Tuple[str, str], float] = {}

    def get_trie_count(self, trie, key: str) -> int:
        """Hàm lấy tần suất từ RecordTrie một cách an toàn"""
        normalized_key = unicodedata.normalize("NFC", key)
        res = trie.get(normalized_key)
        # res sẽ là một list, ví dụ: [(495,)]. Nếu không có key, nó trả về list rỗng []
        return res[0][0] if res else 0

    # MÀNG LỌC ĐỘ DÀI (Tránh rác & Tăng tốc difflib)
    def is_valid_length(self, cand_telex: str, error_len: int) -> bool:
        cand_len = len(cand_telex)
        # Bỏ qua từ quá ngắn (< 3) NẾU từ gốc gõ vào đủ dài (>= 3)
        if error_len >= 3 and cand_len < 3:
            return False
        # Bỏ qua những từ chênh lệch độ dài quá 3 ký tự
        if abs(cand_len - error_len) > 3:
            return False
        return True

    def get_fast_close_matches(
        self, target: str, possibilities: List[str], n: int, cutoff: float
    ) -> List[str]:
        results = []
        for p in possibilities:
            sim = Levenshtein.ratio(target, p)
            if sim >= cutoff:
                results.append((p, sim))

        # Sắp xếp theo độ giống nhau giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:n]]

    # SINH CANDIDATE
    def get_candidates(
        self, error_word: str, prev_word: str | None = None
    ) -> List[str]:
        candidates: List[str] = []

        # Ép chữ lỗi về chuẩn Telex
        error_telex = to_standard_telex(error_word)
        error_len = len(error_telex)

        # Lọc từ Bigram (Những từ từng đi liền sau prev_word)
        if prev_word:
            prefix = f"{prev_word} "
            context_words: List[str] = [
                key[len(prefix) :] for key in self.bigrams.keys(prefix)
            ]

            if context_words:
                # Tạo mapping telex tạm thời cho tập context words
                context_telex_to_word: Dict[str, List[str]] = {}
                for cw in context_words:
                    ct = to_standard_telex(cw)
                    if ct not in context_telex_to_word:
                        context_telex_to_word[ct] = []
                    context_telex_to_word[ct].append(cw)

                # Áp dụng màng lọc độ dài trước khi đưa vào difflib
                context_telex_list = [
                    t
                    for t in context_telex_to_word.keys()
                    if self.is_valid_length(t, error_len)
                ]

                # Fuzzy match trên tập con TELEX
                context_telex_matches: List[str] = self.get_fast_close_matches(
                    error_telex,
                    context_telex_list,
                    n=self.cfg.top_n,
                    cutoff=self.cfg.cutoff,
                )

                # Ánh xạ ngược từ Telex về chữ Tiếng Việt thật
                for ctm in context_telex_matches:
                    for real_word in context_telex_to_word[ctm]:
                        real_word = unicodedata.normalize("NFC", real_word)
                        if real_word not in candidates:
                            candidates.append(real_word)

        # Bổ sung từ Unigram nếu chưa đủ số lượng top_n
        if len(candidates) < self.cfg.top_n:
            filtered_global_telex = [
                t for t in self.telex_vocab_list if self.is_valid_length(t, error_len)
            ]

            # Tìm trên toàn bộ từ điển TELEX
            general_telex_matches: List[str] = self.get_fast_close_matches(
                error_telex,
                filtered_global_telex,
                n=self.cfg.top_n,
                cutoff=self.cfg.cutoff,
            )
            for gtm in general_telex_matches:
                for real_word in self.telex_to_vocab[gtm]:
                    real_word = unicodedata.normalize("NFC", real_word)
                    if real_word not in candidates:
                        candidates.append(real_word)

        return candidates[: self.cfg.top_n]

    def get_kb_cost(self, char1: str, char2: str) -> float:
        # Tính phí phạt khi gõ nhầm char1 thành char2 dựa trên tọa độ bàn phím
        if char1 == char2:
            return 0.0

        if char1 not in self.kb_coords or char2 not in self.kb_coords:
            return self.cfg.unknown_char_penalty

        x1, y1 = self.kb_coords[char1]
        x2, y2 = self.kb_coords[char2]
        # Công thức tính khoảng cách học năm c2, c3 và calculus
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Chuẩn hóa khoảng cách.
        return min(dist / self.cfg.max_kb_distance, 1.0)

    def keyboard_aware_similarity(self, word1: str, word2: str) -> float:
        # Thuật toán Damerau-Levenshtein kết hợp Khoảng cách bàn phím

        # BỘ NHỚ ĐỆM (CACHE): Bỏ qua tính toán nếu đã từng tính cặp từ này
        cache_key = (word1, word2)
        if cache_key in self._sim_cache:
            return self._sim_cache[cache_key]

        m, n = len(word1), len(word2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Khởi tạo ma trận
        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        # Tính toán chi phí biến đổi
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = self.get_kb_cost(word1[i - 1], word2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,  # Deletion (Xóa 1 ký tự)
                    dp[i][j - 1] + 1.0,  # Insertion (Thêm 1 ký tự)
                    dp[i - 1][j - 1] + cost,  # Substitution (Gõ nhầm ký tự)
                )

                # Phép toán đảo vị trí (DAMERAU)
                if (
                    i > 1
                    and j > 1
                    and word1[i - 1] == word2[j - 2]
                    and word1[i - 2] == word2[j - 1]
                ):
                    dp[i][j] = min(
                        dp[i][j], dp[i - 2][j - 2] + self.cfg.transposition_cost
                    )

        max_len = max(m, n)
        if max_len == 0:
            return 1.0

        # Đổi từ 'Khoảng cách' (0.0 -> max_len) sang 'Độ giống nhau' (0.0 -> 1.0)
        distance = dp[m][n]
        sim = math.exp(-distance / max_len)
        final_sim = max(0.0, sim)

        self._sim_cache[cache_key] = final_sim  # Lưu cache
        return max(0.0, final_sim)

    def calculate_context_prob(self, w1: str | None, w2: str, w3: str) -> float:
        """
        Nội suy xác suất kết hợp Trigram, Bigram và Unigram.
        Trả về xác suất chuẩn hóa 0.0 -> 1.0.
        """
        # 1. Tính xác suất Trigram P(w3 | w1, w2)
        p_tri = 0.0
        if w1:
            bigram_w1_w2_count = self.get_trie_count(self.bigrams, f"{w1} {w2}")
            trigram_count = (
                self.get_trie_count(self.trigrams, f"{w1} {w2} {w3}")
                if self.trigrams
                else 0
            )
            p_tri = (
                trigram_count / bigram_w1_w2_count if bigram_w1_w2_count > 0 else 0.0
            )

        # 2. Tính xác suất Bigram P(w3 | w2)
        unigram_w2_count = self.get_trie_count(self.unigrams, w2)
        bigram_w2_w3_count = self.get_trie_count(self.bigrams, f"{w2} {w3}")
        p_bi = bigram_w2_w3_count / unigram_w2_count if unigram_w2_count > 0 else 0.0

        # 3. Tính xác suất Unigram P(w3)
        p_uni = (
            self.get_trie_count(self.unigrams, w3) / self.total_unigrams
            if self.total_unigrams > 0
            else 0.0
        )

        # Trọng số lấy từ cấu hình
        l3 = getattr(self.cfg, "lambda_3", 0.0)
        l2 = getattr(self.cfg, "lambda_2", 0.0)
        l1 = getattr(self.cfg, "lambda_1", 0.0)

        # Nếu chưa đủ 2 từ để có Trigram (vd: mới đến từ thứ 2 trong câu)
        # Ta chia lại tỷ lệ quyền lực cho Bigram và Unigram
        if not w1:
            total_l = l2 + l1
            return (l2 * p_bi + l1 * p_uni) / total_l if total_l > 0 else 0.0

        # Nội suy chuẩn
        return (l3 * p_tri) + (l2 * p_bi) + (l1 * p_uni)

    # TÍNH ĐIỂM
    def calculate_score(
        self,
        candidate: str,
        error_word: str,
        prev_word: str | None,
        prev_prev_word: str | None = None,
    ) -> float:
        # Normalize
        candidate = unicodedata.normalize("NFC", candidate)
        error_word = unicodedata.normalize("NFC", error_word)

        cand_telex = to_standard_telex(candidate)
        err_telex = to_standard_telex(error_word)

        eps = 1e-8

        # Similarity
        sim = self.keyboard_aware_similarity(err_telex, cand_telex)
        sim_feat = math.log(sim + 1e-8)

        # Unigram
        count = self.get_trie_count(self.unigrams, candidate)
        p_uni = count / self.total_unigrams if self.total_unigrams > 0 else 0.0

        # Context (LM)
        if prev_word:
            p_ctx = self.calculate_context_prob(prev_prev_word, prev_word, candidate)

            # Stutter penalty
            if candidate == prev_word:
                p_ctx *= getattr(self.cfg, "stutter_penalty", 0.0)
        else:
            # đầu câu → fallback unigram
            p_ctx = max(p_uni, eps)

        ctx_feat = math.log(p_ctx + eps)
        ctx_feat = (ctx_feat + 10) / 10

        # Weights
        w_sim = getattr(self.cfg, "sim_weight", 0.0)
        w_ctx = getattr(self.cfg, "context_weight", 0.0)

        # Final score (log-linear)
        score = (w_sim * sim_feat) + (w_ctx * ctx_feat)

        # Exact match bonus
        if candidate == error_word and candidate in getattr(
            self, "standard_dict", set()
        ):
            score += getattr(self.cfg, "exact_match_bonus", 1.0)

        # Debug
        if self.debug and self.detail_log:
            prev_str = prev_word if prev_word else "[START]"
            print("ERROR:", err_telex)
            print("CAND :", cand_telex)
            print(f"      ➜ '{prev_str}' -> '{candidate}' (error: '{error_word}')")
            print(f"         sim : {sim_feat:.4f} * {w_sim}")
            print(f"         ctx : {ctx_feat:.4f} * {w_ctx}")
            print(f"         => SCORE: {score:.4f}")

        return score

    def is_delayed_anchor(self, w: str, next_w: str | None) -> bool:
        if not hasattr(self, "standard_dict") or w not in self.standard_dict:
            return False
        if len(w) < 2 or w in getattr(self, "dynamic_ambiguous_words", set()):
            return False
        if next_w is None:
            return True

        # Check Bigram Validation
        if f"{w} {next_w}" in self.bigrams:
            return True
        return False

    # VITERBI DECODING (Sửa lỗi theo ngữ cảnh toàn câu)
    def correct_sentence(self, sentence: str, top_k: int = 5) -> List[str]:
        words: List[str] = sentence.lower().split()
        if not words:
            return []

        # Cấu trúc: { "chuỗi_full_câu": (tổng_điểm, [list_từ], từ_cuối_cùng) }
        paths: Dict[str, Tuple[float, List[str], str]] = {}

        # Biến cờ: Báo hiệu từ tiếp theo phải khởi tạo nhánh mới
        reset_context_next_step = True

        # DUYỆT QUA CÁC TỪ CÒN LẠI
        for i, current_word in enumerate(words):
            new_paths: Dict[str, Tuple[float, List[str], str]] = {}
            is_anchor = False
            is_garbage = False

            if self.debug:
                print(f"\n[VITERBI] Từ thứ {i + 1}: '{current_word}'")

            # LOOKAHEAD: Nhìn trộm từ tiếp theo
            next_word = words[i + 1] if i + 1 < len(words) else None

            # NEO TỪ TRỄ (Delayed Anchor)
            if self.is_delayed_anchor(current_word, next_word):
                candidates = [current_word]
                is_anchor = True
                if self.debug:
                    print(f"  ➜ Đã Neo cứng (Delayed Anchor Passed): '{current_word}'")
            else:
                candidates_set = set()
                if not reset_context_next_step and paths:
                    # Lấy từ cuối cùng của nhánh từ index 2 của Tuple
                    for _, _, prev_cand in paths.values():
                        candidates_set.update(
                            self.get_candidates(current_word, prev_word=prev_cand)
                        )

                candidates_set.update(self.get_candidates(current_word, prev_word=None))
                candidates = list(candidates_set)

                # candidates = candidates[: self.cfg.top_n]

                # BỎ CUỘC
                if candidates:
                    best_sim = max(
                        [Levenshtein.ratio(current_word, c) for c in candidates]
                    )
                    if best_sim < self.cfg.cutoff:
                        candidates = [current_word]
                        is_garbage = True
                        if self.debug:
                            print(
                                f"  ➜ Bỏ cuộc: Rác/Từ lạ (Max Sim {best_sim:.2f} < {self.cfg.cutoff})."
                            )
                else:
                    candidates = [current_word]

            step_log_data: List[Dict] = []

            # TÌM ĐƯỜNG NỐI TỐT NHẤT TỪ BƯỚC TRƯỚC SANG BƯỚC HIỆN TẠI
            for curr_cand in candidates:
                # TRƯỜNG HỢP A: Khởi tạo nhánh mới (Do đứng đầu câu, hoặc đứng sau từ Rác)
                if reset_context_next_step or not paths:
                    # Truyền prev_word = None
                    step_score = self.calculate_score(
                        curr_cand,
                        current_word,
                        prev_word=None,
                        prev_prev_word=None,
                    )

                    # Nếu đang ở giữa câu mà bị reset, ta phải nhặt lại lịch sử đường đi tốt nhất trước đó
                    best_history = []
                    if paths:
                        best_past_key = max(paths.keys(), key=lambda k: paths[k][0])
                        best_history = paths[best_past_key][1]

                    total_score = step_score
                    new_path = best_history + [curr_cand]

                    # Nối chuỗi để làm key phân biệt nhánh
                    new_path_key = " ".join(new_path)
                    new_paths[new_path_key] = (total_score, new_path, curr_cand)

                    if self.debug:
                        step_log_data.append(
                            {
                                "cand": curr_cand,
                                "path": f"[MỚI] -> {curr_cand}",
                                "calc_str": f"(Khởi tạo: {step_score:.4f})",
                                "score": total_score,
                            }
                        )

                # TRƯỜNG HỢP B: Nối tiếp chuỗi Markov bình thường
                else:
                    # Thử nối curr_cand vào tất cả các nhánh (prev_cand) của bước trước
                    for path_key, (prev_score, prev_path, prev_cand) in paths.items():
                        # RÚT TỪ TRƯỚC NỮA TỪ TRONG LỊCH SỬ ĐƯỜNG ĐI
                        # prev_path có dạng: ['hôm', 'nay', 'đi'] -> [-1] là 'đi', [-2] là 'nay'
                        prev_prev_cand = prev_path[-2] if len(prev_path) >= 2 else None

                        # Gọi tính điểm với Trigram Context
                        step_score = self.calculate_score(
                            curr_cand, current_word, prev_cand, prev_prev_cand
                        )

                        # ĐIỂM TÍCH LŨY = Điểm lịch sử + Điểm bước nhảy
                        total_score = prev_score + step_score

                        new_path = prev_path + [curr_cand]
                        # Nối chuỗi để làm key phân biệt nhánh
                        new_path_key = " ".join(new_path)
                        new_paths[new_path_key] = (total_score, new_path, curr_cand)

                        if self.debug:
                            step_log_data.append(
                                {
                                    "cand": curr_cand,
                                    "path": f"{prev_cand} -> {curr_cand}",
                                    "calc_str": f"({prev_score:.4f} + {step_score:.4f})",
                                    "score": total_score,
                                }
                            )

            # Cập nhật các đường đi cho vòng lặp tiếp theo
            paths = new_paths

            # QUYẾT ĐỊNH TRẠNG THÁI CHO TỪ TIẾP THEO
            if is_garbage:
                # Nếu từ hiện tại là rác, NGẮT CHUỖI. Từ tiếp theo sẽ khởi tạo nhánh mới.
                reset_context_next_step = True
            else:
                # Nếu từ hiện tại là Neo cứng hoặc sửa thành công, CHO PHÉP NỐI CHUỖI.
                reset_context_next_step = False

            # BEAM SEARCH PRUNING (TỈA CÀNH)
            # Nếu số lượng nhánh vượt quá beam_width, chỉ giữ lại những nhánh có tổng điểm cao nhất
            if len(paths) > getattr(self.cfg, "beam_width", 5):
                # Sắp xếp paths theo điểm (score nằm ở index 0 của value tuple) giảm dần
                sorted_paths = sorted(
                    paths.items(), key=lambda item: item[1][0], reverse=True
                )
                # Cắt lấy top K
                paths = dict(sorted_paths[: self.cfg.beam_width])

            if self.debug and step_log_data:
                # Sắp xếp theo score từ cao xuống thấp
                step_log_data.sort(key=lambda x: x["score"], reverse=True)

                print("-" * 82)
                print(
                    f"| {'ỨNG VIÊN':<12} | {'TUYẾN TỐT NHẤT':<20} | {'LỊCH SỬ + ĐIỂM BƯỚC NHẢY':<25} | {'TỔNG ĐIỂM':<12} |"
                )
                print("-" * 82)
                for item in step_log_data[:20]:
                    print(
                        f"| {item['cand']:<12} | {item['path']:<20} | {item['calc_str']:<25} | {item['score']:<12.4f} |"
                    )
                print("-" * 82)

        # Chọn ra các tuyến đường cuối cùng có tổng điểm cao nhất
        if paths:
            # Lấy tất cả các giá trị (tổng_điểm, [lịch_sử_câu]) và sắp xếp theo điểm giảm dần
            sorted_paths = sorted(
                paths.values(), key=lambda item: item[0], reverse=True
            )

            results = []
            # Lấy ra tối đa top_k kết quả tốt nhất
            for _, best_sentence, _ in sorted_paths[:top_k]:
                results.append(" ".join(best_sentence))

            return results

        return []
