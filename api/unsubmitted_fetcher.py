"""
Module for fetching unsubmitted alpha data from user-provided URLs.
"""
import requests
import pandas as pd
import time
import json
import logging
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any, Optional
from .auth import check_session_valid

logger = logging.getLogger(__name__)

def fetch_unsubmitted_alphas_from_url(session: requests.Session, url: str) -> List[Dict[str, Any]]:
    """
    Fetch unsubmitted alphas from a user-provided URL.
    
    Args:
        session: The authenticated requests.Session object
        url: User-provided URL for fetching unsubmitted alphas
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an unsubmitted alpha record
    """
    if not check_session_valid(session):
        logger.error("Session is not valid. Please re-authenticate.")
        return []
    
    all_alphas = []
    
    try:
        logger.info(f"Fetching unsubmitted alphas from URL: {url}")
        
        # Parse the URL to extract parameters
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        
        # Convert single-item lists to strings for requests
        clean_params = {}
        for key, value_list in params.items():
            if len(value_list) == 1:
                clean_params[key] = value_list[0]
            else:
                clean_params[key] = value_list
        
        # Make the request using the base URL and extracted parameters
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        
        response = session.get(base_url, params=clean_params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        total_count = data.get('count', 0)
        results = data.get('results', [])
        
        logger.info(f"Fetched {len(results)} unsubmitted alphas from initial request (total available: {total_count})")
        
        # Process the results
        for alpha_data_item in results:
            processed_record = process_unsubmitted_alpha_record(alpha_data_item)
            if processed_record:
                all_alphas.append(processed_record)
        
        # Check if there are more pages to fetch based on limit/offset
        limit = int(clean_params.get('limit', 50))
        offset = int(clean_params.get('offset', 0))
        
        # If there are more records, fetch them in batches
        if total_count > offset + limit:
            logger.info(f"Fetching remaining {total_count - (offset + limit)} unsubmitted alphas in batches...")
            
            current_offset = offset + limit
            while current_offset < total_count:
                batch_params = clean_params.copy()
                batch_params['offset'] = str(current_offset)
                
                try:
                    batch_response = session.get(base_url, params=batch_params, timeout=30)
                    batch_response.raise_for_status()
                    batch_data = batch_response.json()
                    batch_results = batch_data.get('results', [])
                    
                    logger.info(f"Fetched batch at offset {current_offset}: {len(batch_results)} alphas")
                    
                    for alpha_data_item in batch_results:
                        processed_record = process_unsubmitted_alpha_record(alpha_data_item)
                        if processed_record:
                            all_alphas.append(processed_record)
                    
                    current_offset += limit
                    time.sleep(0.5)  # Small delay between requests
                    
                except Exception as e:
                    logger.error(f"Error fetching batch at offset {current_offset}: {e}")
                    break
        
        logger.info(f"Successfully fetched {len(all_alphas)} unsubmitted alphas total")
        return all_alphas
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching unsubmitted alphas from URL {url}: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse unsubmitted alphas response from URL {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching unsubmitted alphas from URL {url}: {e}")
        return []

def process_unsubmitted_alpha_record(alpha_data_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single unsubmitted alpha record from the API response.
    
    Args:
        alpha_data_item: Raw alpha data from API
        
    Returns:
        Processed alpha record or None if processing fails
    """
    try:
        settings = alpha_data_item.get('settings') or {}
        regular_info = alpha_data_item.get('regular') or {}
        is_metrics = alpha_data_item.get('is') or {}
        is_risk_neutralized_metrics = is_metrics.get('riskNeutralized', {})
        
        # For unsubmitted alphas, we don't fetch self_correlation or prod_correlation
        processed_record = {
            'alpha_id': alpha_data_item.get('id'),
            'alpha_type': 'UNSUBMITTED',  # Force type to UNSUBMITTED
            'code': regular_info.get('code'),
            'description': regular_info.get('description'),
            'date_added': alpha_data_item.get('dateCreated'),
            'last_updated': alpha_data_item.get('dateModified'),
            
            'settings_region': settings.get('region'),
            'universe': settings.get('universe'),
            'delay': settings.get('delay'),
            'decay': settings.get('decay'),
            'neutralization': settings.get('neutralization'),
            'settings_truncation': settings.get('truncation'),
            'settings_pasteurization': settings.get('pasteurization'),
            'settings_unit_handling': settings.get('unitHandling'),
            'settings_nan_handling': settings.get('nanHandling'),
            'settings_language': settings.get('language'),
            'settings_max_stock_weight': settings.get('maxStockWeight'),
            'settings_max_group_weight': settings.get('maxGroupWeight'),
            'settings_max_turnover': settings.get('maxTurnover'),

            'is_sharpe': is_metrics.get('sharpe'),
            'is_fitness': is_metrics.get('fitness'),
            'is_returns': is_metrics.get('returns'),
            'is_drawdown': is_metrics.get('drawdown'),
            'is_longcount': is_metrics.get('longCount'),
            'is_shortcount': is_metrics.get('shortCount'),
            'is_turnover': is_metrics.get('turnover'),
            'is_margin': is_metrics.get('margin'),
            'rn_sharpe': is_risk_neutralized_metrics.get('sharpe'),
            'rn_fitness': is_risk_neutralized_metrics.get('fitness'),
        }
        
        # Validate required fields
        if not processed_record['alpha_id'] or not processed_record['settings_region']:
            logger.warning(f"Skipping unsubmitted alpha with missing required fields: {alpha_data_item.get('id')}")
            return None
            
        return processed_record
        
    except Exception as e:
        logger.error(f"Error processing unsubmitted alpha record {alpha_data_item.get('id', 'unknown')}: {e}")
        return None