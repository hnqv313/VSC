# TinyMLS

TinyMLS is not yet Machine Learning-based Spellchecker

## Chạy server

Yêu cầu cài dependency trong `requirements.txt`.

```bash
python main.py
```

Biến môi trường hỗ trợ:

- `HOST`: mặc định `0.0.0.0`
- `PORT`: mặc định `8000`
- `DEBUG`: nhận `1`, `true`, `yes` để bật debug

Ví dụ:

```bash
HOST=127.0.0.1 PORT=8000 python main.py
```

Chế độ chạy:

- `DEBUG=true`: dùng Flask dev server
- mặc định: ưu tiên `waitress` để chạy ổn định lâu dài trên máy

## API

Server đọc cấu hình từ `config.json`. Frontend không truyền path hay file cấu hình lên API.

Base URL mặc định:

```text
http://localhost:8000
```

### `GET /api/health`

Kiểm tra server đang hoạt động.

Ví dụ:

```bash
curl http://localhost:8000/api/health
```

Response:

```json
{
  "status": "ok",
  "checker_loaded": true,
  "build_in_progress": false,
  "active_requests": 0,
  "last_load_error": null
}
```

### `POST /api/check`

Nhận văn bản đầu vào và trả về các gợi ý sửa lỗi.

Request body:

- `text`: chuỗi cần kiểm tra, bắt buộc
- `top_k`: số gợi ý cần trả về, mặc định `5`

Các path cấu hình như `config`, `stats`, `dict` được lấy từ `config.json` phía server, frontend không truyền lên.

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "toi dang go tieng viet",
    "top_k": 3
  }'
```

Response thành công:

```json
{
  "text": "toi dang go tieng viet",
  "top_k": 3,
  "best_correction": "toi đang gõ tiếng việt",
  "processing_ms": 12.4,
  "suggestions": [
    "toi đang gõ tiếng việt",
    "tôi đang gõ tiếng việt",
    "toi đang go tiếng việt"
  ]
}
```

Response lỗi:

```json
{
  "error": "Thiếu trường 'text'."
}
```

### `POST /api/build`

Build lại bộ thống kê ngôn ngữ từ corpus.

Request body:

- `workers`: số worker xử lý, mặc định `1`

`data/` là thư mục input cố định phía server. `stats_path` và `dict_path` được lấy từ `config.json`.

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/build \
  -H "Content-Type: application/json" \
  -d '{
    "workers": 2
  }'
```

Response thành công:

```json
{
  "message": "Xây dựng thống kê hoàn tất.",
  "logs": "..."
}
```

## Tích hợp frontend

Frontend chỉ cần gọi HTTP JSON:

```js
const response = await fetch("http://localhost:8000/api/check", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: userInput,
    top_k: 5,
  }),
});

const data = await response.json();
```

Server đã bật CORS `*`, nên có thể gọi trực tiếp từ frontend chạy domain khác trong môi trường phát triển.

## Ghi chú vận hành

- Checker được preload khi server khởi động để giảm độ trễ ở request đầu tiên.
- Trong lúc build thống kê, request check mới sẽ chờ build hoàn tất rồi mới xử lý.
- API giới hạn input `text` tối đa 2000 ký tự cho mỗi request.
