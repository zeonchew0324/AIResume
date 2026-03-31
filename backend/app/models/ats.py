from pydantic import BaseModel

class ResumeAnalysisResponse(BaseModel):
    score: float
    feedback: list[str]
    suggestions: list[str]
    missing_keywords: list[str] 
