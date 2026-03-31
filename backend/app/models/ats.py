from pydantic import BaseModel

class ResumeAnalysisResponse(BaseModel):
    match_score: float
    feedback: str
    missing_keywords: list[str] 
    suggestions: list[dict]
    
