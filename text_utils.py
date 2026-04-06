from functools import lru_cache


@lru_cache(maxsize=100000)
def to_standard_telex(word: str) -> str:
    # Chuyển đổi một từ tiếng Việt có dấu về dạng gõ Telex tiêu chuẩn.
    word = word.lower()

    telex_map = {
        # Các chữ cái có mũ/râu/đặc biệt
        "ă": "aw",
        "â": "aa",
        "đ": "dd",
        "ê": "ee",
        "ô": "oo",
        "ơ": "ow",
        "ư": "uw",
        # Dấu Sắc (s)
        "á": ("a", "s"),
        "ắ": ("aw", "s"),
        "ấ": ("aa", "s"),
        "é": ("e", "s"),
        "ế": ("ee", "s"),
        "í": ("i", "s"),
        "ó": ("o", "s"),
        "ố": ("oo", "s"),
        "ớ": ("ow", "s"),
        "ú": ("u", "s"),
        "ứ": ("uw", "s"),
        "ý": ("y", "s"),
        # Dấu Huyền (f)
        "à": ("a", "f"),
        "ằ": ("aw", "f"),
        "ầ": ("aa", "f"),
        "è": ("e", "f"),
        "ề": ("ee", "f"),
        "ì": ("i", "f"),
        "ò": ("o", "f"),
        "ồ": ("oo", "f"),
        "ờ": ("ow", "f"),
        "ù": ("u", "f"),
        "ừ": ("uw", "f"),
        "ỳ": ("y", "f"),
        # Dấu Hỏi (r)
        "ả": ("a", "r"),
        "ẳ": ("aw", "r"),
        "ẩ": ("aa", "r"),
        "ẻ": ("e", "r"),
        "ể": ("ee", "r"),
        "ỉ": ("i", "r"),
        "ỏ": ("o", "r"),
        "ổ": ("oo", "r"),
        "ở": ("ow", "r"),
        "ủ": ("u", "r"),
        "ử": ("uw", "r"),
        "ỷ": ("y", "r"),
        # Dấu Ngã (x)
        "ã": ("a", "x"),
        "ẵ": ("aw", "x"),
        "ẫ": ("aa", "x"),
        "ẽ": ("e", "x"),
        "ễ": ("ee", "x"),
        "ĩ": ("i", "x"),
        "õ": ("o", "x"),
        "ỗ": ("oo", "x"),
        "ỡ": ("ow", "x"),
        "ũ": ("u", "x"),
        "ữ": ("uw", "x"),
        "ỹ": ("y", "x"),
        # Dấu Nặng (j)
        "ạ": ("a", "j"),
        "ặ": ("aw", "j"),
        "ậ": ("aa", "j"),
        "ẹ": ("e", "j"),
        "ệ": ("ee", "j"),
        "ị": ("i", "j"),
        "ọ": ("o", "j"),
        "ộ": ("oo", "j"),
        "ợ": ("ow", "j"),
        "ụ": ("u", "j"),
        "ự": ("uw", "j"),
        "ỵ": ("y", "j"),
    }

    base_word = ""
    tone = ""

    for char in word:
        if char in telex_map:
            mapped = telex_map[char]
            if isinstance(mapped, tuple):
                # Nếu là tuple, lấy phần base và cất phần dấu đi
                base_word += mapped[0]
                tone = mapped[1]
            else:
                # Nếu chỉ là chữ mũ/râu (không dấu thanh)
                base_word += mapped
        else:
            # Chữ cái tiếng Anh bình thường (a-z)
            base_word += char

    # Ép dấu xuống cuối cùng của chữ
    return base_word + tone
