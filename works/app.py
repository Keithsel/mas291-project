import requests
import pandas as pd
import time
import datetime
import random
import argparse
import json
import streamlit as st
from streamlit_navigation_bar import st_navbar
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_path)
os.chdir(this_path)
print(os.getcwd())

visitor_id = 'bc3286d7-91ee-4c67-93c3-c43bab418bdd' # Replace with your own visitor ID

def parse_arguments():
    parser = argparse.ArgumentParser(description='Scrape property data from Realtor.com')
    parser.add_argument('args', type=str, help='JSON string of arguments')

    args = parser.parse_args()
    arg_list = json.loads(args.args)

    arg_dict = {
        'city': arg_list[0]['city'],
        'property_type': arg_list[0]['property_type'],
        'listing_status': arg_list[0]['listing_status'],
        'price_min': arg_list[0]['price_min'],
        'price_max': arg_list[0]['price_max'],
        'bedrooms_min': arg_list[0]['bedrooms_min'],
        'bedrooms_max': arg_list[0]['bedrooms_max'],
        'bathrooms_min': arg_list[0]['bathrooms_min'],
        'bathrooms_max': arg_list[0]['bathrooms_max'],
        'save_to': arg_list[0]['save_to'],
        'sold_date_min': arg_list[0].get('sold_date_min', None),
        'sold_date_max': arg_list[0].get('sold_date_max', None)
    }

    return arg_dict

def process_args_to_query(args):
    city_slug = args['city'].replace(' ', '_').replace(',', '').lower()
    query = {
        "search_location": {
            "location": args['city']
        }
    }
    
    if args['listing_status'] == 'Any - for sale':
        query["status"] = ["for_sale", "ready_to_build"]
    elif args['listing_status'] == 'Existing homes':
        query["status"] = ["for_sale"]
    elif args['listing_status'] == 'New construction':
        query["status"] = ["for_sale", "ready_to_build"]
        query["new_construction"] = True
    elif args['listing_status'] == 'Foreclosures':
        query["status"] = ["for_sale", "ready_to_build"]
        query["foreclosure"] = True
    elif args['listing_status'] == 'Sold':
        query["status"] = ["sold"]
        if args['sold_date_min'] and args['sold_date_max']:
            query["sold_date"] = {
                "min": args['sold_date_min'],
                "max": args['sold_date_max']
            }

    if args['price_min'] != 'Any' or args['price_max'] != 'Any':
        query["list_price"] = {}
        if args['price_min'] != 'Any':
            query["list_price"]["min"] = int(args['price_min'])
        if args['price_max'] != 'Any':
            query["list_price"]["max"] = int(args['price_max'])

    if args['bedrooms_min'] != 'None' or args['bedrooms_max'] != 'None':
        query["beds"] = {}
        if args['bedrooms_min'] != 'None':
            query["beds"]["min"] = int(args['bedrooms_min'])
        if args['bedrooms_max'] != 'None':
            query["beds"]["max"] = int(args['bedrooms_max'])

    if args['bathrooms_min'] != 'None' or args['bathrooms_max'] != 'None':
        query["baths"] = {}
        if args['bathrooms_min'] != 'None':
            query["baths"]["min"] = int(args['bathrooms_min'])
        if args['bathrooms_max'] != 'None':
            query["baths"]["max"] = int(args['bathrooms_max'])
    
    if args['property_type'] != 'Any':
        property_types = {
            "House": ["single_family"],
            "Condo": ["condos", "condo_townhome_rowhome_coop", "condo_townhome"],
            "Townhome": ["townhomes", "duplex_triplex", "condo_townhome"],
            "Multi family": ["multi_family"],
            "Mobile": ["mobile"],
            "Farm": ["farm"],
            "Land": ["land"]
        }
        query["type"] = property_types.get(args['property_type'], [args['property_type'].replace(' ', '_').lower()])

    variables = {
        "geoSupportedSlug": city_slug,
        "query": query,
        "client_data": {
            "device_data": {
                "device_type": "desktop"
            }
        },
        "limit": 200,
        "offset": 0,
        "sort_type": "relevant" if args['listing_status'] != 'Sold' else None,
        "sort": [{"field": "sold_date", "direction": "desc"}] if args['listing_status'] == 'Sold' else None,
        "search_promotion": {
            "names": ["CITY"],
            "slots": [],
            "promoted_properties": []
        }
    }
    
    variables = {k: v for k, v in variables.items() if v is not None}

    return {"variables": variables}

def estimate_eta(start_time, current_progress):
    elapsed_time = time.time() - start_time
    if current_progress > 0:
        total_time = elapsed_time / current_progress
        remaining_time = total_time - elapsed_time
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
        return eta.strftime("%H:%M:%S")
    return "Calculating..."

def scrape_properties_with_progress(processed_query):
    url = 'https://www.realtor.com/api/v1/rdc_search_srp?client_id=rdc-search-for-sale-search&schema=vesta'
    headers = {"content-type": "application/json"}
    limit = 200
    all_properties = []
    total_checked = 0
    global_total_properties = 0

    def dynamic_delay(minimum=1.0, maximum=3.0):
        return random.uniform(minimum, maximum)

    def exponential_backoff_retry(request_func, max_retries=5):
        for i in range(max_retries):
            try:
                response = request_func()
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if i == max_retries - 1:
                    st.error(f"Max retries reached. Error: {e}")
                    return None
                wait = (2 ** i) + dynamic_delay(5, 10)
                st.warning(f"Request failed. Retrying in {wait:.2f} seconds...")
                time.sleep(wait)

    def send_request(body):
        return requests.post(url, headers=headers, json=body)
    
    graphql_query = """
        query ConsumerSearchQuery(
            $query: HomeSearchCriteria!
            $limit: Int
            $offset: Int
            $search_promotion: SearchPromotionInput
            $sort: [SearchAPISort]
            $sort_type: SearchSortType
            $client_data: JSON
            $bucket: SearchAPIBucket
            ) {
            home_search: home_search(
                query: $query
                sort: $sort
                limit: $limit
                offset: $offset
                sort_type: $sort_type
                client_data: $client_data
                bucket: $bucket
                search_promotion: $search_promotion
            ) {
                count
                total
                search_promotion {
                names
                slots
                promoted_properties {
                    id
                    from_other_page
                }
                }
                mortgage_params {
                interest_rate
                }
                properties: results {
                property_id
                list_price
                search_promotions {
                    name
                    asset_id
                }
                primary_photo(https: true) {
                    href
                }
                rent_to_own {
                    right_to_purchase
                    rent
                }
                listing_id
                matterport
                virtual_tours {
                    href
                    type
                }
                status
                products {
                    products
                    brand_name
                }
                source {
                    id
                    type
                    spec_id
                    plan_id
                    agents {
                    office_name
                    }
                }
                lead_attributes {
                    show_contact_an_agent
                    opcity_lead_attributes {
                    cashback_enabled
                    flip_the_market_enabled
                    }
                    lead_type
                    ready_connect_mortgage {
                    show_contact_a_lender
                    show_veterans_united
                    }
                }
                community {
                    description {
                    name
                    }
                    property_id
                    permalink
                    advertisers {
                    office {
                        hours
                        phones {
                        type
                        number
                        primary
                        trackable
                        }
                    }
                    }
                    promotions {
                    description
                    href
                    headline
                    }
                }
                permalink
                price_reduced_amount
                description {
                    name
                    beds
                    baths_consolidated
                    sqft
                    lot_sqft
                    baths_max
                    baths_min
                    beds_min
                    beds_max
                    sqft_min
                    sqft_max
                    type
                    sub_type
                    sold_price
                    sold_date
                }
                location {
                    street_view_url
                    address {
                    line
                    postal_code
                    state
                    state_code
                    city
                    coordinate {
                        lat
                        lon
                    }
                    }
                    county {
                    name
                    fips_code
                    }
                }
                open_houses {
                    start_date
                    end_date
                }
                branding {
                    type
                    name
                    photo
                }
                flags {
                    is_coming_soon
                    is_new_listing(days: 14)
                    is_price_reduced(days: 30)
                    is_foreclosure
                    is_new_construction
                    is_pending
                    is_contingent
                }
                list_date
                photos(limit: 2, https: true) {
                    href
                }
                advertisers {
                    type
                    builder {
                    name
                    href
                    logo
                    }
                }
                }
            }
            commute_polygon: get_commute_polygon(query: $query) {
                areas {
                id
                breakpoints {
                    width
                    height
                    zoom
                }
                radius
                center {
                    lat
                    lng
                }
                }
                boundary
            }
            }
    """
    
    body = {
        "query": graphql_query,
        "variables": processed_query["variables"],
        "isClient": True,
        "visitor_id": visitor_id
    }

    city_name = processed_query["variables"]["query"]["search_location"]["location"]
    status = processed_query["variables"]["query"].get("status", ["for_sale", "ready_to_build"])

    overall_progress = st.progress(0)
    overall_status = st.status("Starting scraping process...", expanded=True)
    batch_progress = st.empty()

    stop_button = st.button("Stop Scraping")

    warning_placeholder = st.empty()

    def show_warning(message):
        warning_placeholder.warning(message)
        time.sleep(3)
        warning_placeholder.empty()

    if "sold" in status:
        logger.debug(f"Processed query for sold properties: {processed_query}")
        overall_interval_status = st.status("Preparing to scrape sold properties...", expanded=True)

        earliest_date = datetime.datetime.strptime(processed_query["variables"]["query"]["sold_date"]["min"], "%Y-%m-%d")
        current_date = datetime.datetime.strptime(processed_query["variables"]["query"]["sold_date"]["max"], "%Y-%m-%d")

        def get_global_total_properties():
            nonlocal global_total_properties
            query = processed_query["variables"]["query"].copy()
            body["variables"]["query"] = query

            response = exponential_backoff_retry(lambda: send_request(body))
            if response is not None and 'data' in response.json() and 'home_search' in response.json()['data'] and response.json()['data']['home_search'] is not None:
                data = response.json()
                global_total_properties = data['data']['home_search']['total']
                overall_status.update(label=f"Total sold properties to track for {city_name} in the specified date range: {global_total_properties}")

        get_global_total_properties()
        
        start_time = time.time()
        while current_date >= earliest_date and total_checked < global_total_properties:
            if stop_button:
                show_warning("Scraping stopped by user.")
                break

            start_date = max(current_date - datetime.timedelta(days=365), earliest_date)
            last_total = None

            while True:
                if stop_button:
                    st.warning("Scraping stopped by user.")
                    break

                query = processed_query["variables"]["query"].copy()
                query["sold_date"] = {"min": start_date.strftime("%Y-%m-%d"), "max": current_date.strftime("%Y-%m-%d")}
                body["variables"]["query"] = query

                response = exponential_backoff_retry(lambda: send_request(body))
                if response is not None and 'data' in response.json():
                    logger.debug(f"API response for sold properties: {response.json()}")
                else:
                    logger.error("Invalid response or missing data for sold properties")
                if response is None or 'data' not in response.json() or 'home_search' not in response.json()['data'] or response.json()['data']['home_search'] is None:
                    st.error("Error: Invalid response structure or missing data.")
                    st.json(response.json() if response else "No response")
                    break

                data = response.json()
                api_total = data['data']['home_search']['total']
                overall_interval_status.update(label=f"Properties sold between {start_date.strftime('%Y-%m-%d')} and {current_date.strftime('%Y-%m-%d')}: {api_total}")

                if api_total > 10000:
                    st.warning("More than 10,000 results found. Narrowing the search interval.")
                    start_date += datetime.timedelta(days=30)
                    time.sleep(dynamic_delay(5, 10))
                else:
                    if last_total is None or api_total > last_total:
                        last_total = api_total
                        start_date -= datetime.timedelta(days=1)
                        time.sleep(dynamic_delay(5, 10))
                    else:
                        break

            offset = 0
            batch_progress_bar = batch_progress.progress(0)
            batch_status = st.status(f"Scraping properties sold between {start_date.strftime('%Y-%m-%d')} and {current_date.strftime('%Y-%m-%d')}...", expanded=True)
            
            while offset < api_total:
                if stop_button:
                    show_warning("Scraping stopped by user.")
                    break

                body['variables']['offset'] = offset
                response = exponential_backoff_retry(lambda: send_request(body))
                if response is None:
                    st.error("Failed to fetch data after retries.")
                    break
                data = response.json()
                current_batch = data['data']['home_search']['properties']
                all_properties.extend(current_batch)
                total_checked += len(current_batch)
                offset += limit
                time.sleep(dynamic_delay(2, 3))

                batch_progress_bar.progress(min(offset / api_total, 1.0))
                batch_status.update(label=f"Scraped {offset}/{api_total} properties for current interval")
                
                overall_progress_value = min(total_checked / global_total_properties, 1.0)
                overall_progress.progress(overall_progress_value)

                percentage = overall_progress_value * 100
                eta = estimate_eta(start_time, overall_progress_value)
                overall_status.update(label=f"Overall Progress: {percentage:.2f}% | Properties: {total_checked}/{global_total_properties} | ETA: {eta}")

                if total_checked >= global_total_properties:
                    overall_status.update(label=f"Reached total tracked properties count: {global_total_properties}. Stopping the scraping process.", state="complete")
                    break

            batch_progress.empty()  # Clear the batch progress bar
            current_date = start_date - datetime.timedelta(days=1)
            time.sleep(dynamic_delay(5, 10))
            if total_checked >= global_total_properties or current_date < earliest_date or stop_button:
                break

    else:
        offset = 0
        api_total = None
        first_batch = True
        start_time = time.time()

        while api_total is None or offset < api_total:
            if stop_button:
                show_warning("Scraping stopped by user.")
                break

            body['variables']['offset'] = offset
            response = exponential_backoff_retry(lambda: send_request(body))
            if response is None:
                st.error("Failed to fetch data after retries.")
                break

            data = response.json()

            if api_total is None:
                api_total = data['data']['home_search']['total']
                if first_batch:
                    overall_status.update(label=f"Total properties available for sale in {city_name}: {api_total}")
                    first_batch = False
                    if api_total > 10000:
                        st.warning("More than 10,000 results found. Consider narrowing your search criteria.")
                    
            properties = data['data']['home_search']['properties']
            all_properties.extend(properties)
            total_checked += len(properties)
            offset += limit
            time.sleep(dynamic_delay(2, 3))

            overall_progress_value = min(offset / api_total, 1.0)
            overall_progress.progress(overall_progress_value)

            percentage = overall_progress_value * 100
            eta = estimate_eta(start_time, overall_progress_value)
            overall_status.update(label=f"Progress: {percentage:.2f}% | Properties: {total_checked}/{api_total} | ETA: {eta}")

    if stop_button:
        overall_status.update(label="Scraping process stopped by user", state="error")
    else:
        overall_status.update(label="Scraping process complete", state="complete")

    return {
        "total_properties_checked": total_checked,
        "properties": all_properties
    }

def extract_data(properties):
    extracted_data = []

    for property in properties:
        status = property.get('status', '').lower()
        if 'sold' in status:
            logger.debug(f"Processing sold property: {property.get('property_id')}")

        id = property.get('property_id', None)
        permalink = property.get('permalink', None)
        post_link = "https://www.realtor.com/realestateandhomes-detail/" + permalink if permalink else None
        price = property.get('list_price', None)

        list_date = property.get('list_date', None)
        list_date = list_date.split('T')[0] if list_date else None

        location = property.get('location', {})
        address_line = location.get('address', {}).get('line', None)
        city = location.get('address', {}).get('city', None)
        state_code = location.get('address', {}).get('state_code', None)
        postal_code = location.get('address', {}).get('postal_code', None)
        address = f"{address_line}, {city}, {state_code} {postal_code}" if all([address_line, city, state_code, postal_code]) else None

        status = property.get('status', None)
        status = status.upper() if status else None

        if 'sold' in status.lower():
            sold_date = property.get('description', {}).get('sold_date', None)
            if list_date and sold_date:
                list_date = datetime.datetime.strptime(list_date, "%Y-%m-%d")
                sold_date = datetime.datetime.strptime(sold_date, "%Y-%m-%d")
                days_until_sold = (sold_date - list_date).days
                sold_date = sold_date.strftime("%Y-%m-%d")
                list_date = list_date.strftime("%Y-%m-%d")
            else:
                days_until_sold = None
        else:
            sold_date = "Not sold yet"
            days_until_sold = "Not sold yet"

        description = property.get('description', {})
        area = description.get('sqft', None)
        bedrooms = description.get('beds', None)
        bathrooms = description.get('baths_consolidated', None)

        coordinate = location.get('address', {}).get('coordinate', None)
        latitude = coordinate['lat'] if coordinate else None
        longitude = coordinate['lon'] if coordinate else None

        extracted_data.append({
            'Data Source': 'https://www.realtor.com/',
            'ID': id,
            'Post link': post_link,
            'List date': list_date,
            'Sold date': sold_date,
            'Days until sold': days_until_sold,
            'Price': price,
            'Address': address,
            'Status': status,
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Latitude': latitude,
            'Longitude': longitude
        })

    return extracted_data

def final_touch(extracted_data, sold=False):
    df = pd.DataFrame(extracted_data)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    if sold:
        df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')

        negative_days = df['days_until_sold'] < 0
        df.loc[negative_days, ['list_date', 'sold_date']] = df.loc[negative_days, ['sold_date', 'list_date']].values
        df.loc[negative_days, 'days_until_sold'] = df.loc[negative_days, 'days_until_sold'].abs()
        df['days_until_sold'] = pd.to_numeric(df['days_until_sold'], errors='coerce')

    
    df['status'] = df['status'].str.strip().str.upper()
    df['address'] = df['address'].str.strip().str.title()

    df.dropna(inplace=True)

    return df

def save_data(df, filename):
    df.to_csv(filename, index=True)
    print(f"Data saved to {filename}")

def parsing_data():
    args = parse_arguments()
    processed_query = process_args_to_query(args)
    
    properties_data = scrape_properties_with_progress(processed_query)
    
    extracted_data = extract_data(properties_data["properties"])
    
    df = pd.DataFrame(extracted_data)
    df = final_touch(df, "sold" in processed_query["variables"]["query"].get("status", []))
    
    if args['save_to']:
        save_data(df, args['save_to'])
    else:
        print(df)

if 'queries' not in st.session_state:
    st.session_state.queries = []

def get_all_us_cities():
    df = pd.read_csv("helpers/uscities.csv")
    return [f"{row['city']}, {row['state_id']}" for index, row in df.iterrows()]

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    logo = "works/assets/logo.svg"

    style = {
        "nav": {
            "justify-content": "left",
        },
        "img": {
            "padding-right": "14px",
        },
        "span": {
            "color": "white",
            "padding": "14px",
        },
        "active": {
            "background-color": "#DD5746",
            "color": "white",
            "font-weight": "semibold",
            "padding": "14px",
        },
        "hover": {
            "background-color": "white",
            "color": "black",
            "font-weight": "normal",
            "padding": "14px",
        },
        "ul": {
            "justify-content": "flex-start",
        }
    }
    pages = ["Query Builder", "Results", "EDA"]

    page = st_navbar(
        pages,
        logo_path=logo,
        styles=style
    )
    
    # Display current page
    if page == "Query Builder":
        search_criteria()
    elif page == "Results":
        results_page()
    elif page == "EDA":
        eda_page()
    elif page == "Task 1":
        pass
    elif page == "Task 2":
        pass
    elif page == "Task 3":
        pass

def custom_price_selector(label, options, default, key=None):
    selected_option = st.selectbox(
        label,
        options,
        index=options.index(default),
        key=f"{key}_select" if key else None
    )
    
    if selected_option == "Custom":
        custom_value = st.number_input(
            "Enter value",
            value=0,
            min_value=0,
            step=10000,
            key=f"{key}_custom" if key else None
        )
    else:
        custom_value = None

    return custom_value if selected_option == "Custom" else selected_option

def room_selector(label, default_min, default_max, key=None):
    selected_option = st.selectbox(
        f"{label} Availability",
        ["No", "Yes"],
        index=1 if default_min != "None" or default_max != "None" else 0,
        key=f"{key}_select" if key else None
    )
    
    if selected_option == "Yes":
        min_value = st.selectbox(
            f"{label} Min",
            options=["None"] + list(range(1, 6)),
            index=0 if default_min == "None" else default_min,
            key=f"{key}_min" if key else None
        )
        max_value = st.selectbox(
            f"{label} Max",
            options=["None"] + list(range(1, 6)),
            index=0 if default_max == "None" else default_max,
            key=f"{key}_max" if key else None
        )
    else:
        min_value, max_value = "None", "None"

    return min_value, max_value

def search_criteria():
    st.header("Realtor Scraper")

    with st.container():
        city_col, prop_col, status_col, price_col, room_col = st.columns(5)
        
        with city_col:
            with st.expander("City", expanded=True):
                city_slug_list = get_all_us_cities()
                city = st.selectbox(label="City", options=city_slug_list, index=0, label_visibility="collapsed")

        with prop_col:
            with st.expander("Property Type", expanded=True):
                property_type = st.selectbox("Property Type", ["Any", "House", "Condo", "Townhome", "Multi family", "Mobile", "Farm", "Land"], index=0, label_visibility="collapsed")

        with status_col:
            with st.expander("Listing Status", expanded=True):
                listing_status = st.selectbox("Listing Status", ["Any - for sale", "Existing homes", "New construction", "Foreclosures", "Sold"], index=0, label_visibility="collapsed")
                
                if listing_status == "Sold":
                    sold_date_min = st.date_input(
                        "From",
                        value=datetime.date.today() - datetime.timedelta(days=365),
                        max_value=datetime.date.today()
                    )
                    sold_date_max = st.date_input(
                        "To",
                        value=datetime.date.today(),
                        min_value=sold_date_min,
                        max_value=datetime.date.today()
                    )
                else:
                    sold_date_min, sold_date_max = None, None

        with price_col:
            with st.expander("Price Range", expanded=True):
                price_min_col, price_max_col = st.columns(2)
                with price_min_col:
                    price_min = custom_price_selector("Min Price", ["Any", "Custom"], "Any", key="price_min")
                with price_max_col:
                    price_max = custom_price_selector("Max Price", ["Any", "Custom"], "Any", key="price_max")

        with room_col:
            with st.expander("Rooms", expanded=True):
                col_min, col_max = st.columns(2)
                with col_min:
                    bedrooms_min, bedrooms_max = room_selector("Bedrooms", "None", "None", key="bedrooms")
                with col_max:
                    bathrooms_min, bathrooms_max = room_selector("Bathrooms", "None", "None", key="bathrooms")

    default_save_to = f"data/crawled/{city.split(',')[0].replace(' ', '_').lower()}_data.csv"
    save_to = st.text_input("Save to", value=default_save_to, placeholder="File path for the dataset, default to /data/crawled/city_data.csv")

    if st.button("Scrape"):
        st.session_state.queries = []

        query = {
            "city": city,
            "property_type": property_type,
            "listing_status": listing_status,
            "price_min": price_min,
            "price_max": price_max,
            "bedrooms_min": bedrooms_min,
            "bedrooms_max": bedrooms_max,
            "bathrooms_min": bathrooms_min,
            "bathrooms_max": bathrooms_max,
            "save_to": save_to,
            "sold_date_min": sold_date_min.strftime("%Y-%m-%d") if sold_date_min else None,
            "sold_date_max": sold_date_max.strftime("%Y-%m-%d") if sold_date_max else None
        }
        st.session_state.queries.append(query)
        
        st.subheader("Query Parameters")
        df = pd.DataFrame(st.session_state.queries)
        st.dataframe(df, hide_index=True, use_container_width=True)

        processed_query = process_args_to_query(query)
        properties_data = scrape_properties_with_progress(processed_query)
        
        extracted_data = extract_data(properties_data["properties"])
        
        df = pd.DataFrame(extracted_data)
        df = final_touch(df, "sold" in processed_query["variables"]["query"].get("status", []))
        
        st.session_state.scraped_data = df
        
        if save_to:
            save_data(df, save_to)
            st.success(f"Data saved to {save_to}")
        
        st.success("Scraping complete! View results in the Results page.")

def results_page():
    st.header("Your scraped data")

    with st.expander("View the scraped data, or use your own data for analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use scraped data", key="scraped-data", use_container_width=True):
                st.session_state.which_data = "scraped"
        with col2:
            if st.button("Upload your own data", key="own_data", use_container_width=True):
                st.session_state.which_data = "own"

    if 'which_data' not in st.session_state:
        st.session_state.which_data = ''

    if st.session_state.which_data == "scraped":
        if 'scraped_data' in st.session_state:
            st.session_state.data = st.session_state.scraped_data
            st.dataframe(st.session_state.data, use_container_width=True)
        else:
            st.info("No scraped data available. Please run a scraping operation first.")
    elif st.session_state.which_data == "own":
        uploaded_file = st.file_uploader("Upload your own data", type=["csv"])
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.dataframe(st.session_state.data, use_container_width=True)

def normalize_and_remove_outliers(data, columns_to_normalize):
    data_processed = data.copy()
    for col in columns_to_normalize:
        if col in data_processed.columns:
            # Apply log transformation
            data_processed[col] = np.log1p(data_processed[col])
            
            # Remove outliers
            z_scores = np.abs(stats.zscore(data_processed[col]))
            data_processed = data_processed[z_scores < 3]
    
    return data_processed

def eda_page():
    st.header("Exploratory Data Analysis")

    if 'data' not in st.session_state or st.session_state.data.empty:
        st.info("No data available. Please run a scraping operation or upload your own data in the Results page.")
        return

    data = st.session_state.data

    st.subheader("Data Overview")
    st.dataframe(data.head(), use_container_width=True)

    st.subheader("Data Description")
    st.dataframe(data.describe(), use_container_width=True)

    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("Select Columns for Analysis")
    selected_columns = st.multiselect("Choose columns for analysis", numerical_columns, default=numerical_columns[:4])

    st.subheader("Advanced Options")
    normalize = st.checkbox("Normalize data and remove outliers")
    columns_to_normalize = st.multiselect("Select columns to normalize", selected_columns, default=['price']) if normalize else []

    if st.button("Run EDA"):
        if normalize:
            data_processed = normalize_and_remove_outliers(data, columns_to_normalize)
            st.success(f"Data normalized and outliers removed. Rows remaining: {len(data_processed)}")
        else:
            data_processed = data

        for col in selected_columns:
            st.subheader(f"Analysis for {col}")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Distribution plot
            sns.histplot(data_processed[col], kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {col}")
            
            # Box plot
            sns.boxplot(y=data_processed[col], ax=ax2)
            ax2.set_title(f"Box Plot of {col}")
            
            st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data_processed[selected_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Scatter plot matrix
        st.subheader("Scatter Plot Matrix")
        fig = sns.pairplot(data_processed[selected_columns], height=2.5)
        st.pyplot(fig)

        st.success("EDA completed successfully!")

if __name__ == "__main__":
    main()