from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import dateutil.parser
import re

from graphiti_core.nodes import EpisodeType

class caseParser():

    
    def opinion_to_episode(self, opinion: Dict[str, Any]):
       
        opinion_id = opinion.get('id')
        
        case_name = opinion.get('case_name', 'Unknown Case')
        
        clean_name = case_name.lower()\
            .replace(' v. ', '_v_')\
            .replace(' ', '_')\
            .replace('.', '')\
            .replace(',', '')\
            .replace('\'', '') 
        
        episode_id = f"{clean_name}"
        
        content = opinion.get('plain_text', '')
        if not content:
            raise ValueError(f"Opinion {opinion_id} has no plain_text content")
        
        date = opinion.get('date_filed')
        valid_at = datetime.now(timezone.utc)
        
        if date:
            try:
                valid_at = dateutil.parser.parse(date)
                if valid_at.tzinfo is None:
                    valid_at = valid_at.replace(tzinfo=timezone.utc)
            except Exception as e:
                print(f"Warning: Could not parse date '{date}': {e}")
        
        court = opinion.get('court', 'Unknown Court')
        author = opinion.get('author_str', 'Unknown Author')
        desc = f"Legal Opinion - {case_name} ({court})"
        
        ep = {
            'id': episode_id,
            'content': self.clean_opinion_text(content),
            'type': EpisodeType.text,  
            'description': desc,
            'valid_at': valid_at.date() if hasattr(valid_at, 'date') else valid_at,
            'metadata': {
                'opinion_id': opinion_id,
                'case_name': case_name,
                'court': court,
                'author': author,
                'date_filed': date,
                'docket_number': opinion.get('docket_number', ''),
                'opinion_type': opinion.get('type', 'unknown')
            }
        }
        
        return ep
    
    def clean_opinion_text(self, text: str):

        if not text:
            return ""

        text = text.replace('\x0c', '\n\n')
        text = re.sub(r'^[ ]{3,}', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = text.replace('\xa0', ' ')  
        text = re.sub(r'([^\n])  +', r'\1 ', text)
        text = re.sub(r'\n[ \t]+\n', '\n\n', text) 
        text = text.strip()
        
        return text