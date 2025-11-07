from app import create_app

# Buat instance Flask dari factory function
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
