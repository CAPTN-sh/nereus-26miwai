from requests import get
from requests.exceptions import ReadTimeout, Timeout
from bs4 import BeautifulSoup
from re import match, DOTALL


def crawl_ship_info(mmsi):
    url_base = "http://myshiptracking.com/vessels"
    err, msg, response = get_request(url_base, mmsi)
    if err:
        return err, msg, response

    soup = BeautifulSoup(response.text, "html.parser")
    table = extract_all_table_values(soup)

    data = {}
    data["name"], data["ship_type"] = extract_name_type(soup)
    data["mmsi"] = extract_number(table["MMSI"])
    data["imo"] = extract_number(table["IMO"])
    data["length"], data["width"] = extract_dimension(table["Size"])

    if not data["mmsi"] == mmsi:
        return (1, "MMSI mismatch!", data)

    return (0, "success", data)


def get_request(url_base, mmsi):
    try:
        url = f"{url_base}/{mmsi}"
        response = get(url, timeout=1)
    except (ReadTimeout, Timeout) as err:
        return (1, f"exception: {err}", None)

    if response.status_code in range(400, 600):
        return (1, f"HTTP code: {response.status_code}", None)
    return (0, "success", response)


def extract_all_table_values(soup):
    table = {}
    for th in soup.find_all("th"):
        key = th.get_text(strip=True)
        td = th.find_next("td")
        value = td.get_text(strip=True) if td else None
        table[key] = value
    return table


def extract_name_type(soup):
    title = soup.title
    if title is None:
        return (None, None)

    re = r"(?P<Name>[\w\s/&-]+) - (?P<Type>[\w\s/&-]+) \(.*\) | MyShipTracking"
    m = match(re, title.get_text(), flags=DOTALL)

    if m is None:
        return (None, None)

    return (m.group("Name"), m.group("Type"))


def extract_number(td):
    m = match(r"\d+", td, flags=DOTALL)
    if m is None:
        return None
    return int(m.group())


def extract_dimension(td):
    re = r"(?P<length>\d+) x (?P<width>\d+) m"
    m = match(re, td, flags=DOTALL)

    if m is None:
        return (None, None)

    return (int(m.group("length")), int(m.group("width")))
