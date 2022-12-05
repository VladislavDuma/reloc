import asyncio
import uvicorn
from typing import Union
from fastapi import FastAPI

app = FastAPI()

# TODO: оптимизировать функционал и код

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/func")
async def check_func():
    # res = check_hello()
    # return res
    return {"func": "todo"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


async def main():
    config = uvicorn.Config("server:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
