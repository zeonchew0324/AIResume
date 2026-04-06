from pydantic import BaseModel

class ScoreBreakdown(BaseModel):
    category: str
    score: float
    reason: str 

class ResumeAnalysisResponse(BaseModel):
    match_score: float
    feedback: str
    missing_keywords: list[str] 
    suggestions: list[dict]
    score_breakdown : list[ScoreBreakdown]
    

