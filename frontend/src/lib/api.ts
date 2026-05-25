export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type HealthStatus = {
  status: string;
  dataset_ready: boolean;
  dataset_loading: boolean;
  dataset_error: string | null;
  model_ready: boolean;
  model_loading: boolean;
  model_error: string | null;
  rows: number;
};

export type CommuneSummary = {
  code_insee: string;
  commune: string;
  departement: string | null;
  prix_m2_median: number | null;
  apl_score: number | null;
};

export type CommuneDetail = CommuneSummary & {
  prix_m2_moyen: number | null;
  desert_medical: boolean;
  score_attractivite: number;
  opportunity_score: number;
  population: number | null;
  population_2012: number | null;
  population_2022: number | null;
  population_2024: number | null;
  variation_population_2012_2022_pct: number | null;
  variation_population_2022_2024_pct: number | null;
  densite: number | null;
  nb_ventes: number | null;
  surface_mediane: number | null;
  pct_maisons: number | null;
  is_zrr: boolean | null;
  revenu_median: number | null;
  age_median: number | null;
  taux_chomage: number | null;
  type_zone: string | null;
  latitude: number | null;
  longitude: number | null;
};

export type NearbyCommune = CommuneDetail & {
  distance_km: number | null;
};

export type SmartSearchZone = "Tous" | "Rural" | "Périurbain" | "Urbain";

export type SmartSearchRequest = {
  budget: number;
  surface: number;
  pivot_code_insee?: string | null;
  radius_km?: number | null;
  min_apl?: number | null;
  zone_type: SmartSearchZone;
  require_budget_fit: boolean;
  strict_apl: boolean;
  limit: number;
};

export type SmartSearchResult = {
  rank: number;
  code_insee: string;
  commune: string;
  departement: string | null;
  score: number;
  score_budget: number;
  score_sante: number;
  score_dynamique: number;
  score_marche: number;
  score_distance: number;
  prix_m2_median: number | null;
  estimated_price: number | null;
  budget_delta: number | null;
  budget_fit_ratio: number | null;
  apl_score: number | null;
  desert_medical: boolean;
  population_2024: number | null;
  variation_population_2022_2024_pct: number | null;
  nb_ventes: number | null;
  type_zone: string | null;
  is_zrr: boolean | null;
  distance_km: number | null;
  latitude: number | null;
  longitude: number | null;
  badges: string[];
  reason: string;
};

export type SmartSearchResponse = {
  summary: {
    total_matches: number;
    returned: number;
    budget: number;
    surface: number;
    max_price_m2: number;
    pivot_commune: string | null;
    radius_km: number | null;
    require_budget_fit: boolean;
    radius_ignored?: boolean;
  };
  items: SmartSearchResult[];
};

export type ListingAnalysisResult = {
  commune: string;
  code_insee: string;
  price: number;
  surface: number;
  price_m2: number;
  median_price_m2: number | null;
  delta_pct: number | null;
  verdict: string;
};

export type ListingParseResult = {
  input: string;
  url: string;
  host: string;
  source: string | null;
  supported: boolean;
  listing_id: string | null;
  unit_id: string | null;
  category: string | null;
  commune: string | null;
  commune_slug: string | null;
  code_insee: string | null;
  departement: string | null;
  postal_code: string | null;
  price: number | null;
  surface: number | null;
  warnings: string[];
};

export type ModelImportance = {
  variable: string;
  importance: number;
};

export type ModelSummary = {
  ready: boolean;
  loading: boolean;
  error: string | null;
  summary: null | {
    nom_modele: string;
    features: string[];
    r2: number | null;
    mae: number | null;
    rmse: number | null;
    n: number | null;
    cv_r2_mean: number | null;
    cv_r2_std: number | null;
    cv_mae_mean: number | null;
    cv_mae_std: number | null;
    cv_folds: number | null;
    importance: ModelImportance[];
    comparaison: Array<Record<string, string | number | null>>;
  };
};

export type PredictionFactor = {
  label: string;
  importance: number;
  value: number | string | boolean | null;
};

export type ModelPredictionResult = {
  commune: string;
  code_insee: string;
  surface: number;
  type_bien: string | null;
  announced_price: number | null;
  announced_price_m2: number | null;
  predicted_price_m2: number;
  predicted_price: number;
  range_low: number;
  range_high: number;
  delta_pct: number | null;
  verdict: string;
  confidence: string;
  confidence_note: string;
  factors: PredictionFactor[];
  model: {
    name: string | null;
    mae: number | null;
    r2: number | null;
    n: number | null;
  };
  coverage: string;
};

export type MapMetric = "prix_m2_median" | "apl_score" | "opportunity";

export type MapPoint = {
  code_insee: string;
  commune: string;
  latitude: number;
  longitude: number;
  metric: MapMetric;
  metric_value: number | null;
  prix_m2_median: number | null;
  apl_score: number | null;
  opportunity_score: number;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = `Erreur API ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Keep the HTTP-level message.
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/api/health"),
  searchCommunes: (query: string) =>
    request<{ items: CommuneSummary[] }>(
      `/api/communes?query=${encodeURIComponent(query)}&limit=8`,
    ),
  commune: (codeInsee: string) =>
    request<CommuneDetail>(`/api/communes/${encodeURIComponent(codeInsee)}`),
  nearby: (codeInsee: string) =>
    request<{ items: NearbyCommune[] }>(
      `/api/communes/${encodeURIComponent(codeInsee)}/nearby?limit=4`,
    ),
  smartSearch: (payload: SmartSearchRequest) =>
    request<SmartSearchResponse>("/api/communes/search-smart", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  analyzeListing: (payload: {
    code_insee?: string;
    commune?: string;
    price: number;
    surface: number;
  }) =>
    request<ListingAnalysisResult>("/api/listing/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  parseListing: (url: string) =>
    request<ListingParseResult>("/api/listing/parse", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),
  modelSummary: () => request<ModelSummary>("/api/model/summary"),
  predictModel: (payload: {
    code_insee?: string;
    commune?: string;
    surface: number;
    price?: number;
    type_bien?: string;
  }) =>
    request<ModelPredictionResult>("/api/model/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  refreshModel: () => request<{ accepted: boolean }>("/api/model/refresh", { method: "POST" }),
  map: (metric: MapMetric) =>
    request<{ items: MapPoint[] }>(`/api/map?metric=${metric}&limit=2500`),
};
