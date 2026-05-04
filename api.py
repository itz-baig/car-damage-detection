from fastapi import FastAPI, UploadFile, HTTPException
from improved import analyze_dent
import shutil, uuid, os

app = FastAPI(title="Car Dent Detection API")

@app.get("/")
async def root():
    return {"status": "online", "message": "Car Dent Detection API"}

@app.post("/analyze-dent")
async def analyze_dent_endpoint(file: UploadFile):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded image temporarily
    temp_filename = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join(os.getcwd(), temp_filename)
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Run detection logic
        result = analyze_dent(temp_path)
        
        if "error" in result:
            return {"success": False, "error": result["error"]}
            
        return result

    except Exception as e:
        return {"success": False, "error": str(e)}
        
    finally:
        # Always clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass