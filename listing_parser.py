from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse


SUPPORTED_HOSTS = {
    "leboncoin.fr": "leboncoin",
    "seloger.com": "seloger",
    "selogerneuf.com": "selogerneuf",
    "pap.fr": "pap",
    "bienici.com": "bienici",
}


def _clean_host(hostname: str | None) -> str:
    host = (hostname or "").lower().strip()
    return host[4:] if host.startswith("www.") else host


def _source_from_host(host: str) -> str | None:
    for supported_host, source in SUPPORTED_HOSTS.items():
        if host == supported_host or host.endswith(f".{supported_host}"):
            return source
    return None


def _title_from_slug(slug: str | None) -> str | None:
    if not slug:
        return None
    cleaned = re.sub(r"[-_]+", " ", slug).strip()
    if not cleaned:
        return None
    return " ".join(part.capitalize() for part in cleaned.split())


def _normalize_url(parsed_url) -> str:
    return urlunparse(
        (
            parsed_url.scheme.lower(),
            parsed_url.netloc.lower(),
            parsed_url.path,
            "",
            parsed_url.query,
            "",
        )
    )


def _segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment]


def _parse_leboncoin(segments: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if len(segments) >= 4 and segments[0] == "ad":
        result["category"] = segments[1]
        if segments[-1].isdigit():
            result["listing_id"] = segments[-1]
    else:
        numeric_segments = [segment for segment in segments if segment.isdigit()]
        if numeric_segments:
            result["listing_id"] = numeric_segments[-1]
    return result


def _parse_seloger(segments: list[str], query: dict[str, list[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    numeric_segments = [segment for segment in segments if segment.isdigit()]
    if numeric_segments:
        result["listing_id"] = numeric_segments[-1]

    unit_id = query.get("unitId") or query.get("unitid")
    if unit_id and unit_id[0].isdigit():
        result["unit_id"] = unit_id[0]

    # Typical SeLoger neuf path:
    # /annonces/neuf/programme/amberieu-en-bugey-01/268863029/
    for index, segment in enumerate(segments):
        if segment in {"programme", "appartement", "maison", "bien"} and index + 1 < len(segments):
            city_segment = segments[index + 1]
            match = re.match(r"(?P<city>.+)-(?P<department>\d{2,3})$", city_segment)
            if match:
                result["commune_slug"] = match.group("city")
                result["departement"] = match.group("department")
                result["commune"] = _title_from_slug(match.group("city"))
                break
    return result


def _parse_pap(segments: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    joined = "/".join(segments)
    match = re.search(r"\br(?P<id>\d+)\b", joined)
    if match:
        result["listing_id"] = match.group("id")

    for segment in segments:
        city_match = re.search(r"vente-(?P<city>[a-z0-9-]+?)-(?P<code>\d{5})", segment)
        if city_match:
            result["commune_slug"] = city_match.group("city")
            result["postal_code"] = city_match.group("code")
            result["departement"] = city_match.group("code")[:2]
            result["commune"] = _title_from_slug(city_match.group("city"))
            break
    return result


def _parse_bienici(segments: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    numeric_segments = [segment for segment in segments if segment.isdigit()]
    if numeric_segments:
        result["listing_id"] = numeric_segments[-1]

    for segment in segments:
        match = re.search(r"(?P<city>[a-z0-9-]+)-(?P<code>\d{5})", segment)
        if match:
            result["commune_slug"] = match.group("city")
            result["postal_code"] = match.group("code")
            result["departement"] = match.group("code")[:2]
            result["commune"] = _title_from_slug(match.group("city"))
            break
    return result


def parse_listing_url(url: str) -> dict[str, Any]:
    raw_url = url.strip()
    parsed = urlparse(raw_url if re.match(r"^https?://", raw_url, re.I) else f"https://{raw_url}")
    host = _clean_host(parsed.hostname)
    source = _source_from_host(host)
    segments = _segments(parsed.path)
    query = parse_qs(parsed.query)

    warnings: list[str] = []
    data: dict[str, Any] = {
        "input": url,
        "url": _normalize_url(parsed),
        "host": host,
        "source": source,
        "supported": source is not None,
        "listing_id": None,
        "unit_id": None,
        "category": None,
        "commune": None,
        "commune_slug": None,
        "departement": None,
        "postal_code": None,
        "price": None,
        "surface": None,
        "warnings": warnings,
    }

    if parsed.scheme not in {"http", "https"} or not host:
        data["supported"] = False
        warnings.append("URL invalide.")
        return data

    if source == "leboncoin":
        data.update(_parse_leboncoin(segments))
    elif source in {"seloger", "selogerneuf"}:
        data.update(_parse_seloger(segments, query))
    elif source == "pap":
        data.update(_parse_pap(segments))
    elif source == "bienici":
        data.update(_parse_bienici(segments))
    else:
        warnings.append("Portail non supporté pour le moment.")

    if data["supported"] and not data.get("listing_id"):
        warnings.append("Identifiant d’annonce introuvable dans l’URL.")
    if data["supported"] and not data.get("commune"):
        warnings.append("Commune absente de l’URL, saisie manuelle nécessaire.")
    if data["supported"]:
        warnings.append("Prix et surface ne sont pas extraits sans scraping de la page.")

    return data


def looks_like_listing_url(value: str) -> bool:
    if not value or "." not in value:
        return False
    parsed = urlparse(value if re.match(r"^https?://", value, re.I) else f"https://{value}")
    return _source_from_host(_clean_host(parsed.hostname)) is not None
