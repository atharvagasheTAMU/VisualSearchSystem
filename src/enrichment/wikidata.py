"""
Wikidata SPARQL enrichment for landmarks and places.

Given a landmark/entity name, queries Wikidata for:
  - entity type (Q class)
  - description
  - instance of (landmark, city, monument, stadium, etc.)
  - related tags/categories
"""

import time

import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "PinterestDemo/1.0 (academic project)"

QUERY_TEMPLATE = """
SELECT ?item ?itemLabel ?itemDescription ?instanceLabel WHERE {{
  ?item rdfs:label "{name}"@en.
  OPTIONAL {{ ?item wdt:P31 ?instance. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 3
"""


def query_wikidata(entity_name: str, retries: int = 2) -> list[dict]:
    """
    Query Wikidata SPARQL for an entity name.

    Returns a list of result dicts with keys:
        item, itemLabel, itemDescription, instanceLabel
    """
    query = QUERY_TEMPLATE.format(name=entity_name.replace('"', ""))

    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  Wikidata query failed for '{entity_name}': {e}")
                return []

    return []


def enrich_entity(entity_name: str) -> dict:
    """
    Enrich a landmark/place name with Wikidata information.

    Returns a dict with:
        entity_type, entity_description, entity_tags (list), wikidata_id
    """
    if not entity_name or not isinstance(entity_name, str):
        return {}

    bindings = query_wikidata(entity_name.strip())
    if not bindings:
        return {}

    hit = bindings[0]

    entity_type = None
    entity_tags = set()

    for b in bindings:
        instance = b.get("instanceLabel", {}).get("value")
        if instance:
            entity_tags.add(instance)
            if entity_type is None:
                entity_type = instance

    wikidata_id = hit.get("item", {}).get("value", "").split("/")[-1]
    description = hit.get("itemDescription", {}).get("value", "")

    return {
        "wikidata_id": wikidata_id,
        "entity_type": entity_type,
        "entity_description": description[:300] if description else None,
        "entity_tags": sorted(entity_tags),
    }
