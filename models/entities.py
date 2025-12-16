from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class LegalCase(BaseModel):
    """
    A court case or legal decision with a case name and citation.
    Examples: Martinez v. Garland, INS v. Cardoza-Fonseca, Henriquez-Rivas v. Holder.
    Look for case names with 'v.' and citations like '123 F.3d 456' or '480 U.S. 421'.
    """
    case_name: Optional[str] = Field(None, description="Full name of the case (e.g., 'Smith v. Jones')")
    citation: Optional[str] = Field(None, description="Official citation (e.g., '123 Cal. App. 4th 456')")
    decision_date: Optional[str] = Field(None, description="Date the case was decided")
    court_name: Optional[str] = Field(None, description="Name of the court that decided the case")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction (e.g., 'California', 'Federal')")
    holding: Optional[str] = Field(None, description="The court's holding or decision")


class Court(BaseModel):
    """
    A judicial court that decides cases.
    Examples: United States Supreme Court, Ninth Circuit Court of Appeals, 
    Board of Immigration Appeals (BIA), Immigration Judge.
    Look for phrases like 'Court of Appeals', 'Supreme Court', 'District Court', 'BIA'.
    """
    court_name: Optional[str] = Field(None, description="Official name of the court")
    court_level: Optional[str] = Field(None, description="'supreme', 'appellate', or 'trial'")
    jurisdiction: Optional[str] = Field(None, description="Geographic jurisdiction")


class LegalConcept(BaseModel):
    """
    An abstract legal principle, doctrine, or concept discussed in case law.
    Examples: asylum, persecution, particular social group, due process, 
    well-founded fear, substantial evidence, credibility determination.
    Look for legal terms and principles that are explained or analyzed in the opinion.
    """
    concept_name: Optional[str] = Field(None, description="Name (e.g., 'breach of contract', 'negligence')")
    legal_domain: Optional[str] = Field(None, description="Area of law (contract, tort, property, etc.)")


class LegalIssue(BaseModel):
    """
    A specific legal question or issue the court must decide.
    Examples: 'whether asylum applicant established membership in particular social group',
    'whether evidence was sufficient', 'whether petitioner met burden of proof'.
    Look for phrases like 'the issue is', 'the question presented', 'we must decide whether'.
    """
    issue_text: Optional[str] = Field(None, description="The legal issue or question presented")
    issue_category: Optional[str] = Field(None, description="Category: 'eviction', 'contract breach', 'custody', 'employment'")


class ProceduralPosture(BaseModel):
    """
    The procedural status or stage of the case.
    Examples: petition for review, appeal, motion to dismiss, summary judgment, remand.
    Look for phrases like 'petition for review', 'appeal from', 'motion for', 'remanded', 'affirmed', 'reversed'.
    """
    posture: Optional[str] = Field(None, description="'summary judgment', 'appeal', 'motion to dismiss', 'trial'")
    outcome: Optional[str] = Field(None, description="'granted', 'denied', 'affirmed', 'reversed'")


class StatuteReference(BaseModel):
    """
    A specific statute, law, or code section cited in the case.
    Examples: 8 U.S.C. § 1158, 8 U.S.C. § 1229a, Cal. Civ. Code § 1954.
    Look for citations with format like 'U.S.C. §', 'C.F.R. §', or state code citations.
    """
    statute_citation: Optional[str] = Field(None, description="Statute citation (e.g., 'Cal. Civ. Code § 1954')")
    statute_name: Optional[str] = Field(None, description="Common name of statute")
    uslm_identifier: Optional[str] = Field(None, description="USLM identifier (e.g., '/us/usc/t42/s1983')")
