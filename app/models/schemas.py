from pydantic import BaseModel
from typing import Optional

class ResumeResponse(BaseModel):
    NAME: Optional[str] = None
    EMAIL: Optional[str] = None
    PHONE: Optional[str] = None
    LOCATION: Optional[str] = None
    JOB_TITLE: Optional[str] = None
    COMPANY: Optional[str] = None
    SKILL: Optional[str] = None
    DEGREE: Optional[str] = None
    UNIVERSITY: Optional[str] = None
    CERTIFICATE: Optional[str] = None
    PROJECT: Optional[str] = None
    ADDITIONAL_URLS: Optional[str] = None