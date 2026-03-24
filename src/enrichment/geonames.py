"""
GeoNames REST API wrapper for place name resolution.

Resolves a place name string into structured geographic context:
    city, country, region, latitude, longitude, alternate names.

Requires a free GeoNames account at geonames.org.
Set GEONAMES_USERNAME in your .env file.
"""

import os
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv

load_dotenv()

GEONAMES_BASE_URL = "http://api.geonames.org"
_REQUEST_DELAY = 0.5  # seconds between requests to respect rate limits


class GeoNamesResolver:
    def __init__(self, username: str | None = None, base_url: str = GEONAMES_BASE_URL):
        self.username = username or os.getenv("GEONAMES_USERNAME")
        if not self.username:
            raise EnvironmentError(
                "GEONAMES_USERNAME not set. "
                "Create a free account at geonames.org and add it to .env"
            )
        self.base_url = base_url
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    def resolve_place(self, place_name: str, max_rows: int = 1) -> dict | None:
        """
        Resolve a place name string to structured geographic data.

        Returns a dict with keys:
            city, country, country_code, region, lat, lng, timezone, population
        Returns None if no result found or on API error.
        """
        self._throttle()
        try:
            resp = requests.get(
                f"{self.base_url}/searchJSON",
                params={
                    "q": place_name,
                    "maxRows": max_rows,
                    "username": self.username,
                    "style": "FULL",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  GeoNames API error for '{place_name}': {e}")
            return None

        geonames = data.get("geonames", [])
        if not geonames:
            return None

        hit = geonames[0]
        return {
            "city": hit.get("name"),
            "country": hit.get("countryName"),
            "country_code": hit.get("countryCode"),
            "region": hit.get("adminName1"),
            "lat": hit.get("lat"),
            "lng": hit.get("lng"),
            "timezone": hit.get("timezone", {}).get("timeZoneId"),
            "population": hit.get("population", 0),
            "feature_class": hit.get("fcl"),  # P=city, A=country, L=landmark
            "feature_code": hit.get("fcode"),
        }

    def get_nearby(self, lat: float, lng: float, radius_km: int = 10, max_rows: int = 5) -> list[dict]:
        """Find notable places near a lat/lng coordinate."""
        self._throttle()
        try:
            resp = requests.get(
                f"{self.base_url}/findNearbyPlaceNameJSON",
                params={
                    "lat": lat,
                    "lng": lng,
                    "radius": radius_km,
                    "maxRows": max_rows,
                    "username": self.username,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("geonames", [])
        except requests.RequestException:
            return []


def resolve_place_safe(resolver: GeoNamesResolver, place_name: str) -> dict:
    """
    Resolve a place name, returning an empty dict on any failure.
    Safe to call in batch enrichment loops.
    """
    if not place_name or not isinstance(place_name, str):
        return {}
    result = resolver.resolve_place(place_name.strip())
    return result or {}
