from typing import Literal, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin
import time
import json
import os
import getpass
from pathlib import Path

import pandas as pd

from multiprocessing.pool import ThreadPool
from functools import partial

import tqdm
from legacy.helpful_functions import save_simulation_result, set_alpha_properties, save_pnl, save_yearly_stats, get_alpha_pnl, get_alpha_yearly_stats

DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")

# Define the path to the 'secrets' folder in the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is now legacy/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up one level to project root
SECRETS_DIR = os.path.join(PROJECT_ROOT, "secrets")
COOKIE_FILE_PATH = os.path.join(SECRETS_DIR, "session_cookies.json")
CREDENTIALS_FILE_PATH = os.path.join(SECRETS_DIR, "platform-brain.json")

def save_cookies(session):
    os.makedirs(SECRETS_DIR, exist_ok=True)  # Ensure 'secrets' directory exists
    with open(COOKIE_FILE_PATH, 'w') as f:
        json.dump(session.cookies.get_dict(), f)
    print(f"Cookies saved to {COOKIE_FILE_PATH}")

def load_cookies(session):
    if Path(COOKIE_FILE_PATH).exists():
        with open(COOKIE_FILE_PATH, 'r') as f:
            cookies = json.load(f)
            session.cookies.update(cookies)
        print(f"Cookies loaded from {COOKIE_FILE_PATH}")

def get_credentials():
    """
    Function to get JSON with platform credentials if they exist, or prompts for new ones.
    """
    if Path(CREDENTIALS_FILE_PATH).exists() and os.path.getsize(CREDENTIALS_FILE_PATH) > 2:
        with open(CREDENTIALS_FILE_PATH) as file:
            data = json.load(file)
    else:
        os.makedirs(SECRETS_DIR, exist_ok=True)  # Ensure 'secrets' directory exists
        email = input("Email:\n")
        password = getpass.getpass(prompt="Password:")
        data = {"email": email, "password": password}
        with open(CREDENTIALS_FILE_PATH, "w") as file:
            json.dump(data, file)
    return data["email"], data["password"]

def check_session_timeout(session):
    """
    Function checks if the session is still valid.
    Returns the remaining time before expiration.
    """
    try:
        response = session.get(brain_api_url + "/authentication")
        
        # Handle different response codes
        if response.status_code == 200:
            result = response.json()
            expiry = result.get("token", {}).get("expiry", 0)
            print(f"Session timeout check: {expiry} seconds remaining")
            return expiry
        elif response.status_code == 204:
            # 204 No Content - authentication might be pending
            print("Session check returned 204 (authentication pending)")
            return 0
        elif response.status_code == 401:
            print("Session expired (401 Unauthorized)")
            return 0
        else:
            print(f"Unexpected status code: {response.status_code}")
            return 0
    except json.JSONDecodeError:
        print("Failed to decode JSON response from authentication check")
        return 0
    except Exception as e:
        print(f"Error checking session timeout: {e}")
        return 0

def authenticate_existing_session(s):
    """
    Function to authenticate an existing session object.
    This is useful when you want to use a pre-configured session with custom settings.
    
    Args:
        s: An existing requests.Session object with custom configuration
        
    Returns:
        The authenticated session object
    """
    load_cookies(s)  # Load cookies if available
    
    # Check if the session cookies are still valid
    timeout_remaining = check_session_timeout(s)
    if timeout_remaining > 0:
        print("Reusing existing session...")
        return s
    
    # If the session is not valid, proceed with new authentication
    s.auth = get_credentials()
    r = s.post(brain_api_url + "/authentication")
    
    if r.status_code == requests.status_codes.codes.unauthorized:
        if r.headers.get("WWW-Authenticate") == "persona":
            print(
                "Complete biometrics authentication and press any key to continue: \n"
                + urljoin(r.url, r.headers["Location"]) + "\n"
            )
            input()
            s.post(urljoin(r.url, r.headers["Location"]))
            
            while True:
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input("Biometrics authentication is not complete. Please try again and press any key when completed \n")
                else:
                    break
        else:
            print("\nIncorrect email or password\n")
            os.remove(CREDENTIALS_FILE_PATH)  # Clear saved credentials
            # Use recursion to retry authentication with new credentials
            s.auth = get_credentials()
            return authenticate_existing_session(s)
    
    # After authentication, verify the session is actually valid before saving cookies
    # Wait a bit for authentication to complete
    time.sleep(2)
    
    # Verify the session is truly authenticated
    max_verify_attempts = 3
    for attempt in range(max_verify_attempts):
        timeout_check = check_session_timeout(s)
        if timeout_check > 0:
            print(f"Authentication successful! Session valid for {timeout_check} seconds")
            save_cookies(s)
            return s
        elif attempt < max_verify_attempts - 1:
            print(f"Authentication still pending... waiting (attempt {attempt + 1}/{max_verify_attempts})")
            time.sleep(3)
    
    print("Warning: Authentication completed but session validation is pending. Cookies saved for retry.")
    save_cookies(s)  # Save cookies anyway for next attempt
    return s


def start_session():
    """
    Function to sign in to the platform and return a session object.
    Reuses saved cookies if they are valid.
    """
    s = requests.Session()
    # Configure HTTPAdapter for increased connection pool size
    adapter = HTTPAdapter(pool_connections=250, pool_maxsize=250)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return authenticate_existing_session(s)  # Use the common authentication function


def generate_alpha(
    regular: str = None,
    selection: str = None,
    combo: str = None,
    alpha_type: Literal["REGULAR", "SUPER"] = "REGULAR",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "INDUSTRY",
    truncation: float = 0.08,
    pasteurization: str = "ON",
    maxTrade: str = "OFF",
    test_period: str = "P2Y",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    selection_handling: str = "POSITIVE",
    selection_limit: int = 100,
    visualization: bool = False,
):
    """
    Function generates data to use in simulation
    has default parameters. If alpha_type='REGULAR'
    it generates alpha dictionary using regular input.
    If alpha_type='SUPER'it generates alpha dictionary 
    using selection and combo inputs.
    """
    settings = {
        "instrumentType": "EQUITY",
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "pasteurization": pasteurization,
        "maxTrade" : maxTrade,
        "testPeriod": test_period,
        "unitHandling": unit_handling,
        "nanHandling": nan_handling,
        "language": "FASTEXPR",
        "visualization": visualization,
    }
    if alpha_type == "REGULAR":
        simulation_data = {
            "type": alpha_type,
            "settings": settings,
            "regular": regular,
        }
    elif alpha_type == "SUPER":
        simulation_data = {
            "type": alpha_type,
            "settings": {
                **settings,
                "selectionHandling": selection_handling,
                "selectionLimit": selection_limit,
            },
            "combo": combo,
            "selection": selection
        }
    else:
        print('ERROR: alpha_type should be REGULAR or SUPER')
        return {}
    return simulation_data


def construct_selection_expression(
    selection: str,
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    selection_limit: int = 1000,
    selection_handling: str = 'POSITIVE'):

    selection_data = {
        'settings.instrumentType': instrument_type,
        'settings.region': region,
        'settings.delay': delay,
        'selection': selection,
        'limit': 10,
        'selectionLimit': selection_limit,
        'selectionHandling': selection_handling
    }
    return selection_data


def run_selection(
    s, selection_data: dict
):
    selection_response = s.get(
        brain_api_url + '/simulations/super-selection', params=selection_data
    )
    r = selection_response.json()
    selected_alphas_count = r.get('count')
    message = r.get('message', '')
    return {'selected_alphas_count': selected_alphas_count, 'message': message}


def start_simulation(
    s, simulate_data
):  
    simulate_response = s.post(
        brain_api_url + "/simulations", json=simulate_data
    )
    return simulate_response


def simulation_progress(s,
    simulate_response,
):  

    if simulate_response.status_code // 100 != 2:
        print(simulate_response.text)
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress = s.get(simulation_progress_url)
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:

        print("An error occurred")
        if "message" in simulation_progress.json():
            print(simulation_progress.json()["message"])
        return {"completed": False, "result": {}}

    alpha = simulation_progress.json().get("alpha", 0)
    if alpha == 0:
        return {"completed": False, "result": {}}
    simulation_result = get_simulation_result_json(s, alpha)
    return {"completed": True, "result": simulation_result}



def multisimulation_progress(s,
    simulate_response,
): 
    
    if simulate_response.status_code // 100 != 2:
        print(simulate_response.text)
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress = s.get(simulation_progress_url)
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:
        print("An error occurred")
        if "message" in simulation_progress.json():
            print(simulation_progress.json()["message"])
        return {"completed": False, "result": {}}


    children = simulation_progress.json().get("children", 0)
    if len(children) == 0:
        return {"completed": False, "result": {}}
    children_list = []
    for child in children:
        child_progress = s.get(brain_api_url + "/simulations/" + child)
        alpha = child_progress.json()["alpha"]
        child_result = get_simulation_result_json(s, alpha)
        children_list.append(child_result)
    return {"completed": True, "result": children_list}


def get_prod_corr(s, alpha_id):
    """
    Function gets alpha's production correlation and returns the max value.
    """
    # Make the API request to get the prod correlation for the alpha
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/prod")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    
    # Check if we have any records in the result
    if result.json().get("records", 0) == 0:
        return None  # Return None if there are no records
    
    # Return only the max value from the JSON response
    return result.json().get("max", None)  # Get the max value directly


def check_prod_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's prod_corr test passed
    Saves result to dataframe
    """

    prod_corr_df = get_prod_corr(s, alpha_id)
    value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
    result = [
        {"test": "PROD_CORRELATION", "result": "PASS" if value <= threshold else "FAIL", "limit": threshold, "value": value, "alpha_id": alpha_id}
    ]
    return pd.DataFrame(result)


def get_self_corr(s, alpha_id):
    """
    Function gets alpha's self correlation
    and save result to dataframe
    """

    while True:

        result = s.get(
            brain_api_url + "/alphas/" + alpha_id + "/correlations/self"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    
    records_len = len(result.json()["records"])
    if records_len == 0:
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)

    return self_corr_df


def check_self_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's self_corr test passed
    Saves result to dataframe
    """

    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        result = [{"test": "SELF_CORRELATION", "result": "PASS", "limit": threshold, "value": 0, "alpha_id": alpha_id}]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id
            }
        ]
    return pd.DataFrame(result)



def get_check_submission(s, alpha_id):
    """
    Function gets alpha's check submission checks
    and returns result in dataframe
    """

    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        return pd.DataFrame()
    
    checks_df = pd.DataFrame(
            result.json()["is"]["checks"]
    ).assign(alpha_id=alpha_id)
    
    if 'year' in checks_df:
        ladder_dict = [checks_df.loc[checks_df.index[checks_df['name']=='IS_LADDER_SHARPE']][['value', 'year']].iloc[0].to_dict()]
        checks_df.at[checks_df.index[checks_df['name']=='IS_LADDER_SHARPE'], 'value'] = ladder_dict
        checks_df.drop(['endDate', 'startDate', 'year'], axis=1, inplace=True)

    return checks_df


# def performance_comparison(s, alpha_id, team_id:Optional[str] = None, competition:Optional[str] = None):
#     """
#     Returns performance comparison for merged performance check
#     """
#     if competition is not None:
#         part_url = f'competitions/{competition}'
#     elif team_id is not None:
#         part_url = f'teams/{team_id}'
#     else:
#         part_url = 'users/self'
#     while True:
#         result = s.get(
#             brain_api_url + f"/{part_url}/alphas/" + alpha_id + "/before-and-after-performance"
#         )
#         if "retry-after" in result.headers:
#             time.sleep(float(result.headers["Retry-After"]))
#         else:
#             break
#     if result.json().get("stats", 0) == 0:
#         return {}
#     if result.status_code != 200:
#         return {}

#     return result.json()

def get_new_is_score(s, alpha_id: str, team_id: Optional[str] = None):
    """
    Fetches and returns the before-and-after performance comparison for a specific Alpha.

    Args:
        s (Session): The API session object used to make requests.
        alpha_id (str): The Alpha ID for which performance needs to be checked.
        team_id (Optional[str]): The team ID (optional). If provided, uses team endpoint.
        competition (Optional[str]): The competition name (optional). If provided, uses competition endpoint.

    Returns:
        dict: Performance comparison data, including calculated score differences.
    """
    # Determine the URL part based on inputs
    part_url = f'competitions/ATOM2024'
    # Loop for retry-after header handling
    while True:
        result = s.get(f"{brain_api_url}/{part_url}/alphas/{alpha_id}/before-and-after-performance")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["retry-after"]))
        else:
            break

    # Check if the result is valid and return stats if found
    if result.status_code == 200 and result.json().get("stats", 0) != 0:
        performance_data = result.json()
        
        # Extract before and after score
        before_score = performance_data.get("score", {}).get("before", 0)
        after_score = performance_data.get("score", {}).get("after", 0)
        score_diff = after_score - before_score
        
        # Extract before and after weeklyScore
        before_weekly_score = performance_data.get("weeklyScore", {}).get("before", 0)
        after_weekly_score = performance_data.get("weeklyScore", {}).get("after", 0)
        weekly_score_diff = after_weekly_score - before_weekly_score

        # Return the scores and the differences
        return {
            "before_score": before_score,
            "after_score": after_score,
            "score_diff": score_diff,
            "before_weekly_score": before_weekly_score,
            "after_weekly_score": after_weekly_score,
            "weekly_score_diff": weekly_score_diff
        }
    
    return {}



def submit_alpha(s, alpha_id):
    """
    Function submits an alpha
    This function is not used anywhere
    """
    result = s.post(brain_api_url + "/alphas/" + alpha_id + "/submit")
    while True:
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
            result = s.get(brain_api_url + "/alphas/" + alpha_id + "/submit")
        else:
            break
    return result.status_code == 200


def get_simulation_result_json(s, alpha_id):
    return s.get(brain_api_url + "/alphas/" + alpha_id).json()



def simulate_single_alpha(
    s,
    simulate_data,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()

    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return {'alpha_id': None, 'simulate_data': simulate_data}
    set_alpha_properties(s, simulation_result["result"]["id"])
    return {'alpha_id': simulation_result["result"]["id"], 'simulate_data': simulate_data}


def simulate_multi_alpha(
    s,
    simulate_data_list,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()
    if len(simulate_data_list) == 1:
        return [simulate_single_alpha(s, simulate_data_list[0])]
    simulate_response = start_simulation(s, simulate_data_list)
    simulation_result = multisimulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return [{'alpha_id': None, 'simulate_data': x} for x in simulate_data_list]
    result = [{"alpha_id": x["id"], "simulate_data": {"type": x["type"], "settings": x["settings"], "regular": x["regular"]["code"]}} for x in simulation_result["result"]]
    _ = [set_alpha_properties(s, x["id"]) for x in simulation_result["result"]]
    return result
    

def get_specified_alpha_stats(
    s,
    alpha_id,
    simulate_data,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
):
    """
    Master-Function to get specified in config statistics

    """
    pnl = None
    stats = None

    if alpha_id is None:
        return {'alpha_id': None, 'simulate_data': simulate_data, 'is_stats': None, 'pnl': pnl, 'stats': stats, 'is_tests': None, 'train': None, 'test': None}

    result = get_simulation_result_json(s, alpha_id)
    region = result["settings"]["region"]
    train = result["train"]
    test = result["test"]
    is_stats = pd.DataFrame([{key: value for key, value in result['is'].items() if key!='checks'}]).assign(alpha_id=alpha_id)
    
    if get_pnl:
        pnl = get_alpha_pnl(s, alpha_id)
    if get_stats:
        stats = get_alpha_yearly_stats(s, alpha_id)
    
    if save_result_file:
        save_simulation_result(result)
    if save_pnl_file and get_pnl:
        save_pnl(pnl, alpha_id, region)
    if save_stats_file and get_stats:
        save_yearly_stats(stats, alpha_id, region)

    is_tests = pd.DataFrame(
        result["is"]["checks"]
    ).assign(alpha_id=alpha_id)

    if check_submission:
        is_tests = get_check_submission(s, alpha_id)

        return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests, 'train': train, 'test': test}

    if check_self_corr and not check_submission:
        self_corr_test = check_self_corr_test(s, alpha_id)
        is_tests = (
            is_tests.append(self_corr_test, ignore_index=True, sort=False)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )
    if check_prod_corr and not check_submission:
        prod_corr_test = check_prod_corr_test(s, alpha_id)
        is_tests = (
            is_tests.append(prod_corr_test, ignore_index=True, sort=False)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )

    return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests, 'train': train, 'test': test}


def simulate_alpha_list(
    s,
    alpha_list,
    limit_of_concurrent_simulations=3,
    simulation_config=DEFAULT_CONFIG,
):
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(alpha_list)) as pbar:
            
            for result in pool.imap_unordered(
                partial(simulate_single_alpha, s), alpha_list
            ):
                result_list.append(result)
                pbar.update()

    stats_list_result = []
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(3) as pool:
        for result in pool.map(
            func, result_list
        ):
            stats_list_result.append(result)
    
    return _delete_duplicates_from_result(stats_list_result) 


def simulate_alpha_list_multi(
    s,
    alpha_list,
    limit_of_concurrent_simulations=3,
    limit_of_multi_simulations=3,
    simulation_config=DEFAULT_CONFIG,
):
    if (limit_of_multi_simulations<2) or (limit_of_multi_simulations>10):
        print('Warning, limit of multi-simulation should be 2..10')
        limit_of_multi_simulations = 3
    if len(alpha_list)<10:
        print('Warning, list of alphas too short, single concurrent simulations will be used instead of multisimulations')
        return simulate_alpha_list(s, alpha_list, simulation_config=simulation_config)
    if any(d['type'] == 'SUPER' for d in alpha_list):
        print('Warning, multisimulation is not supported for SuperAlphas')
        return simulate_alpha_list(s, alpha_list, limit_of_concurrent_simulations=3, simulation_config=simulation_config)
    
    tasks = [alpha_list[i:i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)]
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(tasks)) as pbar:
                
            for result in pool.imap_unordered(
                partial(simulate_multi_alpha, s), tasks
            ):
                result_list.append(result)
                pbar.update()
    result_list_flat = [item for sublist in result_list for item in sublist]
    
    stats_list_result = []
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(3) as pool:
        for result in pool.map(
            func, result_list_flat
        ):
            stats_list_result.append(result)
            
    return _delete_duplicates_from_result(stats_list_result) 


def _delete_duplicates_from_result(result):
    alpha_id_lst = []
    result_new = []
    for x in result:
        if x['alpha_id'] is not None:
            if x['alpha_id'] not in alpha_id_lst:
                result_new.append(x)
                alpha_id_lst.append(x['alpha_id'])
        else:
            result_new.append(x)
    return result_new


def main():
    """
    Main function
    """

    s = start_session()

    k = [
        "vwap * 2",
        "open * close",
        "high * low",
        "vwap * 3",
        "open * close",
        "high * low",
    ]
    alpha_list = [generate_alpha(x) for x in k]

    simulate_alpha_list(s, alpha_list)


if __name__ == "__main__":
    main()
