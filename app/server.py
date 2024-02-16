from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")



from gemini_functions_agent import agent_executor as gemini_functions_agent_chain

add_routes(app, gemini_functions_agent_chain, path="/gemini-functions-agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
