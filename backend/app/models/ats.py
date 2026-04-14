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

class ResumeImprovementChange(BaseModel):
    section: str
    change: str
    
class ResumeImprovementResponse(BaseModel):
    improved_resume: str
    changes: list[ResumeImprovementChange]
    keywords_added: list[str]

class CoverLetterResponse(BaseModel):
    cover_letter: str