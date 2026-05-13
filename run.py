import uvicorn

if __name__ == "__main__":
    print("Khởi động AI Resume Parser Server...")
    # Chạy ứng dụng FastAPI ở cổng 8000
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)