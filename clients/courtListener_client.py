
import os
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, parse_qs
import time

class courtListenerClient:
    
    BASE_URL = "https://www.courtlistener.com/api/rest/v4/"
    
    def __init__(self, api_token: Optional[str] = None):
        
        self.api_token = api_token 
        if not self.api_token:
            raise ValueError(
                "API token required. Set COURTLISTENER_API_KEY env var "
                "variable or pass api_token paramr."
            )
        
        self.headers = {
            "Authorization": f"Token {self.api_token}",
            "Accept": "application/json"
        }

    def _makeRequest(self,endpoint: str,params: Optional[Dict[str, Any]] = None):
        
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            res = requests.get(url, headers=self.headers, params=params)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise

    
    def fetchOpinions(self,  search_query: str, max_res: int = 20, order_by='score desc', skip_results=0, date_start=None, date_end=None):
        
        print(f"Searching for: '{search_query}'")
        print(f"Skipping {skip_results} Cases")

        params = {"q": search_query,"type": "o"}
        
        if date_start:
            params["filed_after"] = date_start
        if date_end:  
            params["filed_before"] = date_end

        search_res = self._makeRequest("search/", params=params)
        print(f"Found {search_res['count']} resuls")
        
        full_ops = []
        num_taken = 0
        page_num = 1
        
        
        while num_taken < max_res and search_res:
            results = search_res.get('results', [])
            
            if not results:
                print("No more results available")
                break
            
            print(f"\nProcessing page {page_num}")
           
            for result in results:
                if num_taken >= max_res:
                    break  

                for op_data in result.get('opinions', []):
                    if num_taken >= max_res:
                        break

                    if skip_results>0:
                        skip_results-=1
                        print("Skipping Opinion")
                        continue
                    
                    opid = op_data['id']
                    
                    try:
                       
                        full_text = self._makeRequest(f"opinions/{opid}/", params={
                            "fields": "id,plain_text,html,author_str,type"
                        })
                        
                        
                        if full_text.get('plain_text') and full_text['plain_text'].strip():
                            full_text['case_name'] = result['caseName']
                            full_text['court'] = result['court']
                            full_text['date_filed'] = result['dateFiled']
                            full_text['docket_number'] = result.get('docketNumber', '')
                            
                            full_ops.append(full_text)
                            num_taken += 1
                            print(f"{num_taken}/{max_res}: {result['caseName'][:50]} : {result['dateFiled']}")
                        else:
                            print(f" Skipping opinion {opid} - no plain text")
                            
                    except Exception as e:
                        print(f" Error fetching opinion {opid}: {e}")
                        if hasattr(e, 'response') and e.response.status_code == 429:
                            sleep = 2000
                            for remaining in range(sleep, 0, -1):
                                mins, secs = divmod(remaining, 60)
                                timer = f"{mins:02d}:{secs:02d}"
                                print(f"\rTime remaining: {timer}  ", end='', flush=True)
                                time.sleep(1)
                            
                            print(f"\rCooldown complete!\n")
                        continue
                    
           
            if num_taken < max_res:
                next_url = search_res.get('next')
                
                if next_url:
                    page_num += 1
                    
                
                    parsed = urlparse(next_url)
                    params = {k: v[0] if len(v) == 1 else v 
                             for k, v in parse_qs(parsed.query).items()}
                    
                    search_res = self._makeRequest("search/", params=params)
                else:
                    break
                
        print(f"\nCollected {len(full_ops)} opinions with plain text")
        return full_ops


