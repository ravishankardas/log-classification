import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from classify import classify

app = FastAPI()

@app.post("/api/v1/classify/csv/")
async def classify_logs(file: UploadFile):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    try:
        df = pd.read_csv(file.file)
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.post("/api/v1/classify/single/")
async def classify_logs_single(source: str, log_message: str):
    if not source or not log_message:
        raise HTTPException(status_code=400, detail="Source and log_message must be provided.")
    try:
        df = pd.DataFrame({"source": [source], "log_message": [log_message]})
        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))