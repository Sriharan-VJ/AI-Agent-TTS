from fastapi import FastAPI
from routers.kokoro_router import router
import os
import uvicorn

app = FastAPI(title="Kokoro Text To Speech")

os.makedirs("output", exist_ok=True)

app.include_router(router, tags=["Kokoro"])


def main():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8998,
        log_level="info",
        reload=False,
        use_colors=True,
    )


if __name__ == "__main__":
    print(
        """
██╗░░██╗░█████╗░██╗░░██╗░█████╗░██████╗░░█████╗░  ████████╗████████╗░██████╗
██║░██╔╝██╔══██╗██║░██╔╝██╔══██╗██╔══██╗██╔══██╗  ╚══██╔══╝╚══██╔══╝██╔════╝
█████═╝░██║░░██║█████═╝░██║░░██║██████╔╝██║░░██║  ░░░██║░░░░░░██║░░░╚█████╗░
██╔═██╗░██║░░██║██╔═██╗░██║░░██║██╔══██╗██║░░██║  ░░░██║░░░░░░██║░░░░╚═══██╗
██║░╚██╗╚█████╔╝██║░╚██╗╚█████╔╝██║░░██║╚█████╔╝  ░░░██║░░░░░░██║░░░██████╔╝
╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░╚════╝░  ░░░╚═╝░░░░░░╚═╝░░░╚═════╝░"""
    )
    main()
