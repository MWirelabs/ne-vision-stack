from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import shutil
import os

app = FastAPI()

# Load DocTR detector + NE-OCR recognizer
model = ocr_predictor(
    det_arch='db_resnet50',
    reco_arch='MWirelabs/ne-ocr',
    pretrained=True
)

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc = DocumentFile.from_images(temp_path)
    result = model(doc)

    os.remove(temp_path)

    extracted_text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                extracted_text += line.render() + "\n"

    return JSONResponse(content={"text": extracted_text})