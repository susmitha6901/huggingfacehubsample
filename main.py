from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import os
from pdf import create_vectorstore_from_pdf, get_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:8080"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask-pdf/")

async def ask_pdf(
    files: list[UploadFile] = File(...),
    question: str = Form(...),
    api_key: str = Form(...)
):
    try:
        # Save multiple PDFs to temporary files
        temp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                contents = await file.read()
                tmp.write(contents)
                temp_paths.append(tmp.name)

        # Vectorstore creation & QA
        vectorstore = create_vectorstore_from_pdf(temp_paths, api_key)
        answer = get_answer(vectorstore, question, api_key)

        # Cleanup
        for path in temp_paths:
            os.remove(path)

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
