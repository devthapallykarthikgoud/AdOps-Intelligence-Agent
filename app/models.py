from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Severity(str, Enum):
    OK = "OK"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Campaign:
    campaign_id: str
    campaign_name: str
    impressions: float
    clicks: float
    revenue: float
    fill_rate: float
    ctr: float = 0.0
    cpm: float = 0.0


@dataclass
class Issue:
    issue_type: str
    metric_value: float
    threshold: float
    description: str


@dataclass
class CampaignAnalysis:
    campaign_id: str
    campaign_name: str
    ctr: float
    cpm: float
    fill_rate: float
    severity: Severity
    issues: List[Issue] = field(default_factory=list)
    rag_context: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)