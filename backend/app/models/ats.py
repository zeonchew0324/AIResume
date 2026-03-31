from pydantic import BaseModel

class ResumeAnalysisResponse(BaseModel):
    match_score: float
    feedback: list[str]
    missing_keywords: list[str] 
    suggestions: list[str]
    
