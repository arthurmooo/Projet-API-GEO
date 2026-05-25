"use client";

import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import type { MapMetric, MapPoint } from "@/lib/api";
import { useEffect } from "react";

type CommuneMapProps = {
  points: MapPoint[];
  selectedCode?: string;
  metric: MapMetric;
  onSelect: (codeInsee: string) => void;
};

const metricLabels: Record<MapMetric, string> = {
  prix_m2_median: "Prix/m²",
  apl_score: "APL",
  opportunity: "Opportunité",
};

function colorFor(point: MapPoint, metric: MapMetric) {
  const value = point.metric_value ?? 0;
  if (metric === "apl_score") {
    if (value < 2.5) return "#d86b3d";
    if (value < 3.5) return "#f0c85f";
    return "#2f7444";
  }
  if (metric === "opportunity") {
    if (value >= 75) return "#225b35";
    if (value >= 55) return "#f0c85f";
    return "#d86b3d";
  }
  if (value > 3500) return "#d86b3d";
  if (value > 2500) return "#f4a35d";
  if (value > 1800) return "#f0c85f";
  if (value > 1300) return "#b7d49d";
  return "#5f9962";
}

function FitSelected({ points, selectedCode }: Pick<CommuneMapProps, "points" | "selectedCode">) {
  const map = useMap();

  useEffect(() => {
    const selected = points.find((point) => point.code_insee === selectedCode);
    if (selected) {
      map.flyTo([selected.latitude, selected.longitude], 10, { duration: 0.8 });
      return;
    }

    if (points.length > 0) {
      const center: LatLngExpression = [
        points.reduce((sum, point) => sum + point.latitude, 0) / points.length,
        points.reduce((sum, point) => sum + point.longitude, 0) / points.length,
      ];
      map.setView(center, 6);
    }
  }, [map, points, selectedCode]);

  return null;
}

export default function CommuneMap({ points, selectedCode, metric, onSelect }: CommuneMapProps) {
  const selected = points.find((point) => point.code_insee === selectedCode);
  const center: LatLngExpression = selected
    ? [selected.latitude, selected.longitude]
    : points[0]
      ? [points[0].latitude, points[0].longitude]
      : [46.7, 2.5];

  return (
    <MapContainer className="osmMap" center={center} zoom={selected ? 10 : 6} scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitSelected points={points} selectedCode={selectedCode} />
      {points.map((point) => {
        const isSelected = point.code_insee === selectedCode;
        return (
          <CircleMarker
            key={point.code_insee}
            center={[point.latitude, point.longitude]}
            radius={isSelected ? 10 : 6}
            pathOptions={{
              color: isSelected ? "#123b22" : "#ffffff",
              fillColor: colorFor(point, metric),
              fillOpacity: isSelected ? 0.95 : 0.78,
              opacity: 1,
              weight: isSelected ? 3 : 1,
            }}
            eventHandlers={{ click: () => onSelect(point.code_insee) }}
          >
            <Tooltip direction="top" offset={[0, -5]}>
              {point.commune}
            </Tooltip>
            <Popup>
              <strong>{point.commune}</strong>
              <br />
              {metricLabels[metric]} : {point.metric_value ?? "n/a"}
              <br />
              Prix médian : {point.prix_m2_median ?? "n/a"} €/m²
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
