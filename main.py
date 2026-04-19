import os

from api import create_app

app = create_app()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}
    if debug:
        app.run(host=host, port=port, debug=True)
        return

    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port, debug=False)
        return

    serve(app, host=host, port=port)


if __name__ == "__main__":
    main()
