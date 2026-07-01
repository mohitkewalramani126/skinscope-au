def health() -> dict:
    """Basic health check for the SkinScope service."""
    return {"status": "ok", "service": "skinscope-au"}


if __name__ == "__main__":
    print(health())