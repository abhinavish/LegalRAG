from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class CitesCase(BaseModel):
    """A relationship where one legal case cites another legal case as legal precedent or authority.
    
    EXAMPLES:
    - "See Henriquez-Rivas v. Holder, 707 F.3d 1081" -> CitesCase with 'cites' type
    - "Following INS v. Cardoza-Fonseca, 480 U.S. 421 (1987)" -> CitesCase with 'follows' type
    - "We OVERRULE Martinez v. Garland" -> CitesCase with 'overrules' type
    - "We distinguish Smith v. Jones" -> CitesCase with 'distinguishes' type
    
    Extract the relationship_type based on keywords: follows, distinguishes, overrules, or cites.
    Extract citation_context from the full sentence containing the case citation.
    """
    
    relationship_type: str = Field(
        None,
        description="REQUIRED: Extract from text. Must be one of: 'follows' (when the opinion agrees with or adopts the cited case's reasoning), 'distinguishes' (when the opinion differentiates from the cited case), 'overrules' (when the opinion explicitly overturns or rejects the cited case), or 'cites' (for general references). Look for keywords like 'following', 'overrule', 'distinguish', 'See'."
    )
    citation_context: str = Field(
        None,
        description="REQUIRED: Extract the complete sentence where this case is cited. This should be the exact text from the opinion that contains the case name."
    )


class DecidedBy(BaseModel):
    """A relationship where a legal case was decided by a specific court.
    
    EXAMPLES:
    - "United States Court of Appeals for the Ninth Circuit"-> DecidedBy
    - "The Supreme Court held that" -> DecidedBy
    - "Board of Immigration Appeals (BIA) affirmed" -> DecidedBy
    - "The Immigration Judge (IJ) denied all forms of relief" -> DecidedBy
    
    This links cases to the courts that issued the decisions.
    """
    name: str = Field(default="DecidedBy", description="Edge type name")
    decision_date: Optional[datetime] = Field(
        None, 
        description="The date when the court issued the decision (format: YYYY-MM-DD)"
    )


class InvolvesConcept(BaseModel):
    """A relationship between a legal case and an abstract legal principle, doctrine, or concept.
    
    EXAMPLES:
    - "To qualify for asylum, an applicant must demonstrate..." -> InvolvesConcept (asylum)
    - "We review under the substantial evidence standard" -> InvolvesConcept (substantial evidence)
    - "Martinez failed to establish membership in a particular social group" -> InvolvesConcept (particular social group)
    - "The court conducted a credibility determination" -> InvolvesConcept (credibility determination)
    
    This connects cases to the legal concepts they analyze or apply.
    """
    name: str = Field(default="InvolvesConcept", description="Edge type name")
    relevance: Optional[str] = Field(
        None, 
        description="Centrality of the concept: 'primary' (main issue), 'secondary' (supporting issue), or 'mentioned' (referenced briefly)"
    )


class Addresses(BaseModel):
    """A relationship where a legal case addresses or resolves a specific legal question or issue.
    
    EXAMPLES:
    - "The issue is whether Martinez established membership in a particular social group" -> Addresses
    - "We must decide whether the BIA's decision was supported by substantial evidence" -> Addresses
    - "Martinez argues that young Salvadoran males who refuse to join gangs constitute a cognizable particular social group" -> Addresses
    - "The question is whether the Immigration Judge properly assessed credibility" -> Addresses
    
    This captures the specific legal questions the court must resolve.
    """
    name: str = Field(default="Addresses", description="Edge type name")
    centrality: Optional[str] = Field(
        None, 
        description="Importance to the case: 'primary' (dispositive issue), 'secondary' (important but not dispositive), 'mentioned' (referenced but not decided)"
    )


class HasPosture(BaseModel):
    """A relationship between a case and its procedural status or stage in the judicial process.
    
    EXAMPLES:
    - "This petition for review concerns..." -> HasPosture (petition for review)
    - "The petition for review is GRANTED" -> HasPosture (granted)
    - "The case is REMANDED to the BIA for further proceedings" -> HasPosture (remanded)
    - "The BIA affirmed without opinion" -> HasPosture (affirmed)
    - "Martinez appeals the Immigration Judge's denial" -> HasPosture (appeal)
    
    This tracks the procedural history and current status of the case.
    """
    name: str = Field(default="HasPosture", description="Edge type name")
    stage: Optional[str] = Field(
        None, 
        description="Stage of litigation: 'trial' (initial proceeding), 'appeal' (intermediate appellate review), or 'supreme court' (highest appellate review)"
    )


class ReferencesStatute(BaseModel):
    """A relationship where a legal case cites, interprets, or applies a specific statute or regulation.
    
    EXAMPLES:
    - "Martinez applied for asylum under 8 U.S.C. § 1158" -> ReferencesStatute
    - "To qualify for asylum, an applicant must demonstrate... 8 U.S.C. § 1101(a)(42)(A)" -> ReferencesStatute
    - "removal proceedings under 8 U.S.C. § 1229a" -> ReferencesStatute
    - "withholding of removal under 8 U.S.C. § 1231(b)(3)" -> ReferencesStatute
    
    This links cases to the statutory provisions they interpret or apply.
    """
    name: str = Field(default="ReferencesStatute", description="Edge type name")
    interpretation_type: Optional[str] = Field(
        None, 
        description="How the case uses the statute: 'interprets' (explains meaning), 'applies' (uses to decide case), or 'challenges' (questions validity/constitutionality)"
    )
