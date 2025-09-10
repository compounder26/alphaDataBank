import pandas as pd
import os
import time
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")
brain_url = os.environ.get("BRAIN_URL", "https://platform.worldquantbrain.com")

class OffsetLimitExceeded(Exception):
    """Raised when API offset limit (10,000) is exceeded."""
    pass

def make_clickable_alpha_id(alpha_id):
    """
    Make alpha_id clickable in dataframes
    So you can go to the platform to analyze simulation result
    """

    url = brain_url + "/alpha/"
    return f'<a href="{url}{alpha_id}">{alpha_id}</a>'


def prettify_result(
    result, detailed_tests_view=False, clickable_alpha_id: bool = False
):
    """
    Combine needed results in one dataframe to analyze your alphas
    Sort by fitness absolute value
    """
    list_of_is_stats = [
        result[x]["is_stats"]
        for x in range(len(result))
        if result[x]["is_stats"] is not None
    ]
    is_stats_df = pd.concat(list_of_is_stats).reset_index(drop=True)
    is_stats_df = is_stats_df.sort_values("fitness", ascending=False)

    expressions = {
        result[x]["alpha_id"]: {"selection": result[x]["simulate_data"]["selection"],
                                "combo": result[x]["simulate_data"]["combo"]}
        if result[x]["simulate_data"]["type"] == "SUPER" 
        else result[x]["simulate_data"]["regular"]
        for x in range(len(result))
        if result[x]["is_stats"] is not None
        }
    expression_df = pd.DataFrame(
        list(expressions.items()), columns=["alpha_id", "expression"]
    )

    list_of_is_tests = [
        result[x]["is_tests"]
        for x in range(len(result))
        if result[x]["is_tests"] is not None
    ]
    is_tests_df = pd.concat(list_of_is_tests, sort=True).reset_index(drop=True)
    is_tests_df = is_tests_df[is_tests_df["result"] != "WARNING"]
    if detailed_tests_view:
        cols = ["limit", "result", "value"]
        is_tests_df["details"] = is_tests_df[cols].to_dict(orient="records")
        is_tests_df = is_tests_df.pivot(
            index="alpha_id", columns="name", values="details"
        ).reset_index()
    else:
        is_tests_df = is_tests_df.pivot(
            index="alpha_id", columns="name", values="result"
        ).reset_index()

    alpha_stats = pd.merge(is_stats_df, expression_df, on="alpha_id")
    alpha_stats = pd.merge(alpha_stats, is_tests_df, on="alpha_id")
    alpha_stats = alpha_stats.drop(
        columns=alpha_stats.columns[(alpha_stats == "PENDING").any()]
    )
    alpha_stats.columns = alpha_stats.columns.str.replace(
        "(?<=[a-z])(?=[A-Z])", "_", regex=True
    ).str.lower()
    if clickable_alpha_id:
        return alpha_stats.style.format({"alpha_id": make_clickable_alpha_id})
    return alpha_stats


def concat_pnl(result):
    """
    Combine needed results in one dataframe to analyze pnls of your alphas
    """
    list_of_pnls = [
        result[x]["pnl"]
        for x in range(len(result))
        if result[x]["pnl"] is not None
    ]
    pnls_df = pd.concat(list_of_pnls).reset_index()

    return pnls_df


def concat_is_tests(result):
    is_tests_list = [
        result[x]["is_tests"]
        for x in range(len(result))
        if result[x]["is_tests"] is not None
    ]
    is_tests_df = pd.concat(is_tests_list, sort=True).reset_index(drop=True)
    return is_tests_df


def save_simulation_result(result):
    """
    Dump simulation result to folder simulation_results
    to json file
    """

    alpha_id = result["id"]
    region = result["settings"]["region"]
    folder_path = "simulation_results/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}")

    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(result, file)

def set_alpha_properties(
    s,
    alpha_id,
    name: str = None,
    color: str = None,
    selection_desc: str = "None",
    combo_desc: str = "None",
    tags: str = ["ace_tag"],
):
    """
    Function changes alpha's description parameters
    """

    params = {
        "color": color,
        "name": name,
        "tags": tags,
        "category": None,
        "regular": {"description": None},
        "combo": {"description": combo_desc},
        "selection": {"description": selection_desc},
    }
    response = s.patch(
        brain_api_url + "/alphas/" + alpha_id, json=params
    )



def save_pnl(pnl_df, alpha_id, region):
    """
    Dump pnl to folder alphas_pnl
    to csv file
    """

    folder_path = "alphas_pnl/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}")
    os.makedirs(folder_path, exist_ok=True)

    pnl_df.to_csv(file_path)


def save_yearly_stats(yearly_stats, alpha_id, region):
    """
    Dump yearly-stats to folder yearly_stats
    to csv file
    """

    folder_path = "yearly_stats/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}")
    os.makedirs(folder_path, exist_ok=True)    

    yearly_stats.to_csv(file_path, index=False)


def get_alpha_pnl(s, alpha_id):
    """
    Function gets alpha pnl of simulation
    """

    while True:
        result = s.get(
            brain_api_url + "/alphas/" + alpha_id + "/recordsets/pnl"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    pnl = result.json().get("records", 0)
    if pnl == 0:
        return pd.DataFrame()
    if len(pnl[0]) == 2:
        pnl_df = (
            pd.DataFrame(pnl, columns=["Date", "Pnl"])
            .assign(
                alpha_id=alpha_id, Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
            )
            .set_index("Date")
        )
    if len(pnl[0]) == 3:
        
        pnl_df = (
            pd.DataFrame(pnl, columns=["Date", "Pnl_combo", "Pnl_eqw"])
            .assign(
                alpha_id=alpha_id, Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
            )
            .set_index("Date")
        )
    return pnl_df


def get_alpha_yearly_stats(s, alpha_id):
    """
    Function gets yearly-stats of simulation
    """

    while True:
        result = s.get(
            brain_api_url + "/alphas/"
            + alpha_id
            + "/recordsets/yearly-stats"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    stats = result.json()
    
    if stats.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in stats["schema"]["properties"]]
    yearly_stats_df = pd.DataFrame(stats["records"], columns=columns).assign(alpha_id=alpha_id)
    return yearly_stats_df

def get_datasets(
    s,
    instrument_type: str = 'EQUITY',
    region: str = 'EUR',
    delay: int = 1,
    universe: str = 'TOP2500'
):
    url = brain_api_url + "/data-sets?" +\
        f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}"
    result = s.get(url)
    datasets_df = pd.DataFrame(result.json()['results'])
    return datasets_df

def _fetch_datafield_batch(
    session: requests.Session, 
    url_template: str,
    offset: int,
    limit: int = 50,
    max_backoff_seconds: int = 300,
    retry_wait_seconds: int = 15
) -> List[Dict[str, Any]]:
    """
    Fetch a single batch of datafields with indefinite retries for transient errors.
    
    Args:
        session: The authenticated session object
        url_template: URL template with {x} placeholder for offset
        offset: The offset for this batch
        limit: Batch size (default 50)
        max_backoff_seconds: Maximum backoff time (default 300 seconds = 5 minutes)
        retry_wait_seconds: Base retry wait time (default 15 seconds)
        
    Returns:
        List of datafield dictionaries, or empty list on permanent failure
    """
    attempt = 0
    while True:  # Indefinite retry loop for transient errors
        attempt += 1
        try:
            logger.debug(f"Fetching datafields batch at offset {offset}, attempt {attempt}")
            url = url_template.format(x=offset)
            
            # Use adaptive timeout that increases with retry attempts
            base_timeout = 20
            timeout_seconds = min(base_timeout + (attempt * 10), 120)  # Max 2 minutes
            if attempt > 1:
                logger.debug(f"Using adaptive timeout ({timeout_seconds}s) for attempt {attempt}")
            
            response = session.get(url, timeout=timeout_seconds)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "results" not in data:
                        logger.warning(f"No 'results' key in response for offset {offset}")
                        return []
                    
                    results = data.get("results", [])
                    logger.debug(f"Successfully fetched {len(results)} datafields at offset {offset}")
                    return results
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for offset {offset}: {e}. Response text: {response.text[:200]}")
                    # Fall through to retry logic
                    
            elif response.status_code in [401, 403]:
                logger.error(f"Authentication error ({response.status_code}) for offset {offset}. Stopping retries.")
                return []  # Stop retrying on auth errors
                
            elif response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limit (429) hit for offset {offset}. Waiting 60 seconds...")
                time.sleep(60)  # Longer wait for rate limits
                continue  # Skip default wait, go to next iteration
                
            elif response.status_code == 400:
                # Check for specific offset limit error
                error_text = response.text.lower()
                if "invalid offset" in error_text or "narrow down" in error_text:
                    logger.warning(f"API offset limit reached at offset {offset}. "
                                 f"Response: {response.text[:200]}")
                    return []  # Signal end of pagination - no more retries
                else:
                    logger.warning(f"Bad request (400) for offset {offset}: {response.text[:200]}")
                    # Fall through to retry logic for other 400 errors
                    
            else:
                logger.warning(f"Request failed with status {response.status_code} for offset {offset}: {response.text[:200]}")
                # Fall through to retry logic
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for offset {offset}")
            # Fall through to retry logic
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error for offset {offset}: {e}")
            # Fall through to retry logic
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request exception for offset {offset}: {e}")
            # Fall through to retry logic
            
        except KeyboardInterrupt:
            logger.info(f"Keyboard interrupt received for offset {offset}. Stopping.")
            raise  # Re-raise KeyboardInterrupt
            
        # Universal retry limit - give up after reasonable attempts regardless of universe
        if attempt >= 20:  # Reasonable retry limit for any universe
            logger.warning(f"Giving up after {attempt} attempts for offset {offset}")
            logger.debug(f"This may indicate either no data available or persistent connectivity issues")
            return []
        
        # Exponential backoff with cap for retries
        wait_duration = min(retry_wait_seconds * min(attempt, 8), max_backoff_seconds)
        logger.info(f"Retrying datafields fetch for offset {offset} in {wait_duration}s (attempt {attempt})")
        try:
            time.sleep(wait_duration)
        except KeyboardInterrupt:
            logger.info(f"Keyboard interrupt during retry wait for offset {offset}. Stopping.")
            raise

def _fetch_dataset_datafields(
    s,
    dataset_id: str,
    instrument_type: str,
    region: str,
    delay: int,
    universe: str,
    theme: str,
    data_type: str,
    search: str,
) -> tuple[str, pd.DataFrame, Optional[str]]:
    """
    Fetch datafields for a single dataset - used for parallel execution.
    
    Args:
        s: Authenticated session object
        dataset_id: ID of the dataset to fetch
        instrument_type: Type of instrument (default "EQUITY")
        region: Region code (e.g., "USA", "EUR") 
        delay: Delay value (0 or 1)
        universe: Universe code (e.g., "TOP3000")
        theme: Theme parameter (default "false")
        data_type: Data type (e.g., "MATRIX", "VECTOR")
        search: Search string filter
        
    Returns:
        Tuple of (dataset_id, dataframe, error_message)
    """
    try:
        logger.debug(f"Fetching datafields for dataset {dataset_id}")
        
        dataset_df = _get_datafields_with_offset_pagination(
            s, instrument_type, region, delay, universe, theme, 
            dataset_id, data_type, search
        )
        
        if not dataset_df.empty:
            logger.debug(f"Dataset {dataset_id}: {len(dataset_df)} datafields")
            return dataset_id, dataset_df, None
        else:
            logger.debug(f"Dataset {dataset_id}: No datafields found")
            return dataset_id, pd.DataFrame(), None
            
    except Exception as e:
        logger.warning(f"Failed to fetch datafields for dataset {dataset_id}: {e}")
        return dataset_id, pd.DataFrame(), str(e)


def _get_datafields_primary(
    s,
    instrument_type: str = "EQUITY",
    region: str = "ASI",
    delay: int = 1,
    universe: str = "MINVOL1M",
    theme: str = "false",
    dataset_id: str = "",
    data_type: str = "VECTOR",
    search: str = "",
) -> pd.DataFrame:
    """
    Primary strategy to retrieve datafields using dataset-based approach.
    
    This method avoids the 10,000 offset limit by fetching available datasets
    for the given parameters and then fetching datafields for each dataset
    individually. This ensures complete data collection without pagination limits.

    Args:
        s: An authenticated session object.
        instrument_type (str, optional): The type of instrument. Defaults to "EQUITY".
        region (str, optional): The region. Defaults to "ASI".
        delay (int, optional): The delay. Defaults to 1.
        universe (str, optional): The universe. Defaults to "MINVOL1M".
        theme (str, optional): The theme. Defaults to "false".
        dataset_id (str, optional): The ID of a specific dataset. If provided, fetches only this dataset.
        data_type (str, optional): The type of data. Defaults to "VECTOR".
        search (str, optional): A search string to filter datafields. Defaults to "".

    Returns:
        pandas.DataFrame: A DataFrame containing information about available datafields.
    """
    combo_name = f"{data_type}/{region}/{universe}/delay_{delay}"
    
    # If a specific dataset_id is provided, use the old offset-based approach for that dataset
    # since dataset-specific queries are typically small and won't hit the limit
    if dataset_id:
        logger.info(f"Fetching datafields for specific dataset {dataset_id} in {combo_name}")
        return _get_datafields_with_offset_pagination(s, instrument_type, region, delay, 
                                                    universe, theme, dataset_id, data_type, search)
    
    # Use dataset-based approach for comprehensive fetching
    logger.info(f"Using dataset-based approach for {combo_name}")
    
    # Get available datasets for this combination
    datasets = get_available_datasets(s, region, universe, delay, instrument_type)
    
    if not datasets:
        logger.warning(f"No datasets found for {combo_name} - returning empty DataFrame")
        return pd.DataFrame()
    
    logger.info(f"Found {len(datasets)} datasets for {combo_name}, fetching datafields...")
    
    all_datafields = []
    successful_datasets = 0
    failed_datasets = 0
    
    # Determine max workers (cap at 20 parallel requests to avoid overwhelming API)
    max_workers = min(20, len(datasets))
    logger.info(f"Fetching datafields from {len(datasets)} datasets using {max_workers} workers...")
    
    # Use ThreadPoolExecutor for parallel dataset fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all dataset fetch tasks
        future_to_dataset = {
            executor.submit(_fetch_dataset_datafields, s, dataset, instrument_type, 
                           region, delay, universe, theme, data_type, search): dataset
            for dataset in datasets
        }
        
        # Process results as they complete
        for future in as_completed(future_to_dataset):
            try:
                dataset_id, df, error = future.result()
                
                if error:
                    failed_datasets += 1
                    logger.debug(f"Dataset {dataset_id} failed: {error}")
                elif not df.empty:
                    all_datafields.append(df)
                    successful_datasets += 1
                    logger.debug(f"Dataset {dataset_id}: {len(df)} datafields")
                else:
                    logger.debug(f"Dataset {dataset_id}: No datafields found")
                    
            except Exception as e:
                failed_datasets += 1
                logger.warning(f"Unexpected error processing dataset result: {e}")
                continue
    
    logger.info(f"Dataset-based fetch completed for {combo_name}: "
               f"{successful_datasets} successful, {failed_datasets} failed")
    
    if not all_datafields:
        logger.warning(f"No datafields collected from any dataset for {combo_name}")
        return pd.DataFrame()
    
    # Combine all datasets
    logger.debug(f"Combining datafields from {len(all_datafields)} datasets...")
    combined_df = pd.concat(all_datafields, ignore_index=True)
    
    # Remove duplicates based on 'id' column  
    initial_count = len(combined_df)
    if 'id' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['id']).reset_index(drop=True)
        final_count = len(combined_df)
        duplicates_removed = initial_count - final_count
        
        logger.info(f"Dataset-based result for {combo_name}: "
                   f"{final_count} unique datafields ({duplicates_removed} duplicates removed)")
    else:
        logger.warning(f"No 'id' column found for deduplication in {combo_name}")
        final_count = len(combined_df)
        logger.info(f"Dataset-based result for {combo_name}: {final_count} datafields")
    
    return combined_df


def _get_datafields_with_offset_pagination(
    s,
    instrument_type: str = "EQUITY",
    region: str = "ASI",
    delay: int = 1,
    universe: str = "MINVOL1M",
    theme: str = "false",
    dataset_id: str = "",
    data_type: str = "VECTOR",
    search: str = "",
) -> pd.DataFrame:
    """
    Legacy offset-based pagination approach for dataset-specific queries.
    
    This function is only used for fetching datafields from specific datasets,
    which typically have small result sets that won't hit the 10,000 limit.
    """
    limit = 50
    type_param = f"&type={data_type}" if data_type != "ALL" else ""
    
    # Build URL template
    base_url = (
        brain_api_url + "/data-fields?"
        + f"&instrumentType={instrument_type}"
        + f"&region={region}&delay={str(delay)}&universe={universe}{type_param}"
        + f"&dataset.id={dataset_id}&limit={limit}"
    )
    
    if search:
        url_template = base_url + f"&search={search}" + "&offset={x}"
    else:
        url_template = base_url + "&offset={x}"
    
    # Dynamic pagination - fetch until we get a partial or empty batch
    all_datafields = []
    offset = 0
    
    try:
        while True:
            # Use our robust batch fetcher
            batch_results = _fetch_datafield_batch(s, url_template, offset, limit)
            
            if not batch_results:
                # Empty batch - we're done
                break
            
            all_datafields.extend(batch_results)
            
            if len(batch_results) < limit:
                # Partial batch - this is the last page
                break
            
            offset += limit
            
    except KeyboardInterrupt:
        logger.info(f"Keyboard interrupt received during offset pagination.")
        raise
    
    if not all_datafields:
        return pd.DataFrame()
    
    # Convert to DataFrame and expand dictionary columns
    datafields_df = pd.DataFrame(all_datafields)
    datafields_df = expand_dict_columns(datafields_df)
    
    return datafields_df


def get_datafields(
    s,
    instrument_type: str = "EQUITY",
    region: str = "ASI", 
    delay: int = 1,
    universe: str = "MINVOL1M",
    theme: str = "false",
    dataset_id: str = "",
    data_type: str = "VECTOR",
    search: str = "",
) -> pd.DataFrame:
    """
    Retrieve datafields using dataset-based approach to avoid offset limits.
    
    This function uses a pure dataset-based strategy that avoids the 10,000 
    offset limit by fetching datasets first and then querying datafields per dataset.
    This ensures complete data collection for all combinations.
    
    Args:
        s: An authenticated session object.
        instrument_type (str, optional): The type of instrument. Defaults to "EQUITY".
        region (str, optional): The region. Defaults to "ASI". 
        delay (int, optional): The delay. Defaults to 1.
        universe (str, optional): The universe. Defaults to "MINVOL1M".
        theme (str, optional): The theme. Defaults to "false".
        dataset_id (str, optional): The ID of a specific dataset. Defaults to "".
        data_type (str, optional): The type of data. Defaults to "VECTOR".
        search (str, optional): A search string to filter datafields. Defaults to "".
        
    Returns:
        pandas.DataFrame: A DataFrame containing information about available datafields.
    """
    return _get_datafields_primary(s, instrument_type, region, delay, universe, 
                                 theme, dataset_id, data_type, search)


def expand_dict_columns(df):
    """
    Expand dictionary columns in a DataFrame into separate columns.
    
    Args:
        df (pd.DataFrame): DataFrame with potential dictionary columns
        
    Returns:
        pd.DataFrame: DataFrame with expanded dictionary columns
    """
    if df.empty:
        return df
    
    result_df = df.copy()
    
    # Common columns that might contain dictionaries
    dict_columns_to_expand = ['dataset', 'category', 'subcategory']
    
    for col in dict_columns_to_expand:
        if col in result_df.columns:
            try:
                # Check if column contains dictionaries
                sample_value = result_df[col].dropna().iloc[0] if not result_df[col].dropna().empty else None
                if isinstance(sample_value, dict):
                    # Expand dictionary into separate columns
                    expanded = pd.json_normalize(result_df[col].dropna())
                    # Add prefix to avoid column name conflicts
                    expanded.columns = [f"{col}_{subcol}" for subcol in expanded.columns]
                    # Merge with original DataFrame
                    expanded.index = result_df[col].dropna().index
                    result_df = result_df.join(expanded, how='left')
            except (IndexError, TypeError, AttributeError):
                # If expansion fails, keep original column
                pass
    
    return result_df


def get_available_datasets(session, region: str, universe: str, delay: int, 
                          instrument_type: str = "EQUITY") -> List[str]:
    """
    Get list of available dataset IDs for given parameters.
    
    This function fetches available datasets for a specific region/universe/delay
    combination and returns their IDs. Used as a fallback strategy when offset
    pagination hits the 10,000 limit.
    
    Args:
        session: Authenticated session object
        region: Region filter (e.g., 'USA', 'EUR')  
        universe: Universe filter (e.g., 'TOP3000', 'TOP1000')
        delay: Delay filter (0 or 1)
        instrument_type: Instrument type (default 'EQUITY')
        
    Returns:
        List of dataset ID strings, or empty list if none found
    """
    try:
        logger.debug(f"Fetching available datasets for {region}/{universe}/delay_{delay}")
        
        # Use the existing get_datasets function
        datasets_df = get_datasets(session, instrument_type, region, delay, universe)
        
        if not datasets_df.empty and 'id' in datasets_df.columns:
            dataset_ids = datasets_df['id'].unique().tolist()
            logger.debug(f"Found {len(dataset_ids)} datasets for {region}/{universe}/delay_{delay}")
            return dataset_ids
        else:
            logger.debug(f"No datasets found or missing 'id' column for {region}/{universe}/delay_{delay}")
            return []
            
    except Exception as e:
        logger.warning(f"Could not fetch datasets for {region}/{universe}/delay_{delay}: {e}")
        return []


