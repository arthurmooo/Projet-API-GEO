"use client";

import dynamic from "next/dynamic";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type CommuneDetail,
  type CommuneSummary,
  type HealthStatus,
  type ListingParseResult,
  type MapMetric,
  type MapPoint,
  type ModelPredictionResult,
  type ModelSummary,
  type NearbyCommune,
} from "@/lib/api";
import SmartSearch from "./SmartSearch";

const CommuneMap = dynamic(() => import("./CommuneMap"), {
  ssr: false,
  loading: () => <div className="mapLoading">Chargement de la carte...</div>,
});

type View = "home" | "search" | "listing" | "agent" | "market" | "model";

const navItems: Array<{ id: View; label: string }> = [
  { id: "home", label: "Accueil" },
  { id: "search", label: "Recherche" },
  { id: "listing", label: "Annonce" },
  { id: "market", label: "Marché local" },
  { id: "model", label: "Modèle" },
];

function formatNumber(value?: number | null, maximumFractionDigits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits }).format(value);
}

function formatEuro(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${formatNumber(value)} €`;
}

function looksLikeUrl(value: string) {
  return /^https?:\/\//i.test(value.trim()) || /\b(leboncoin|seloger|selogerneuf|pap|bienici)\./i.test(value);
}

function TopNav({
  activeView,
  health,
  onNavigate,
}: {
  activeView: View;
  health: HealthStatus | null;
  onNavigate: (view: View) => void;
}) {
  return (
    <header className="topbar">
      <button className="brand brandButton" type="button" onClick={() => onNavigate("home")} aria-label="Territoire Immo">
        <span className="brandMark" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span>Territoire Immo</span>
      </button>

      <nav className="tabs" aria-label="Navigation principale">
        {navItems.map((item) => (
          <button
            className={activeView === item.id ? "tab active" : "tab"}
            type="button"
            key={item.id}
            onClick={() => onNavigate(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <button className="exportButton" type="button" onClick={() => onNavigate("model")}>
        <span aria-hidden="true">ML</span>
        {health?.model_ready ? "Modèle prêt" : health?.model_loading ? "Entraînement" : "Modèle"}
        {health?.model_loading ? <i className="statusDot" aria-label="Modèle en cours" /> : null}
      </button>
    </header>
  );
}

function SearchBox({
  query,
  suggestions,
  showSuggestions,
  loading,
  placeholder,
  onQueryChange,
  onSelect,
  onFocus,
  onBlur,
}: {
  query: string;
  suggestions: CommuneSummary[];
  showSuggestions: boolean;
  loading: boolean;
  placeholder: string;
  onQueryChange: (value: string) => void;
  onSelect: (codeInsee: string) => void;
  onFocus: () => void;
  onBlur: () => void;
}) {
  const shouldShowDropdown = showSuggestions && (suggestions.length > 0 || loading || query.trim().length > 0);

  return (
    <div className="searchField compactSearch">
      <label className="srOnly" htmlFor="commune-search">Commune</label>
      <span className="searchIcon" aria-hidden="true" />
      <input
        id="commune-search"
        aria-label="Commune"
        aria-autocomplete="list"
        aria-controls="commune-suggestions"
        aria-expanded={shouldShowDropdown}
        autoComplete="off"
        placeholder={placeholder}
        role="combobox"
        value={query}
        onBlur={onBlur}
        onChange={(event) => onQueryChange(event.target.value)}
        onFocus={onFocus}
      />
      {shouldShowDropdown ? (
        <div className="suggestions" id="commune-suggestions" aria-label="Communes trouvées" role="listbox">
          {suggestions.map((item) => (
            <button
              type="button"
              key={item.code_insee}
              onClick={() => onSelect(item.code_insee)}
              onMouseDown={(event) => event.preventDefault()}
              role="option"
              aria-selected={false}
            >
              <strong>{item.commune}</strong>
              <span>{item.code_insee}</span>
              <small>{formatEuro(item.prix_m2_median)}/m²</small>
            </button>
          ))}
          {loading ? <div className="suggestionState">Recherche...</div> : null}
          {!loading && suggestions.length === 0 && query.trim() ? <div className="suggestionState">Aucune commune trouvée</div> : null}
        </div>
      ) : null}
    </div>
  );
}

function HomeHero({ onNavigate }: { onNavigate: (view: View) => void }) {
  return (
    <section className="productHero" aria-labelledby="hero-title">
      <div className="coveragePill">Prototype ML - département 01</div>
      <h1 id="hero-title">Estimer un prix immobilier avec un modèle, pas seulement lire un dataset.</h1>
      <p>
        Territoire Immo transforme les ventes DVF, les signaux de commune et l’accès aux soins en décision lisible :
        communes compatibles, prix cohérent, ou estimation argumentée.
      </p>

      <div className="journeyGrid">
        <button className="journeyCard primaryJourney" type="button" onClick={() => onNavigate("search")}>
          <span>Acheteur</span>
          <strong>Je cherche où acheter</strong>
          <small>Classez les communes selon votre budget, votre surface cible, le rayon et l’accès aux soins.</small>
        </button>
        <button className="journeyCard" type="button" onClick={() => onNavigate("listing")}>
          <span>Annonce</span>
          <strong>J’évalue un prix</strong>
          <small>Collez un lien ou saisissez prix, surface et commune pour voir si le prix est cohérent.</small>
        </button>
      </div>
    </section>
  );
}

function PredictionResult({
  prediction,
  mode,
  onShowModel,
}: {
  prediction: ModelPredictionResult | null;
  mode: "listing" | "agent";
  onShowModel: () => void;
}) {
  if (!prediction) {
    return (
      <aside className="predictionEmpty">
        <span>Décision ML</span>
        <strong>Le résultat apparaîtra ici.</strong>
        <p>Le modèle combine les indicateurs locaux disponibles pour prédire un prix au m², puis compare cette estimation au prix annoncé si vous l’avez renseigné.</p>
      </aside>
    );
  }

  const delta = prediction.delta_pct;
  const deltaLabel = delta === null ? "Comparaison non renseignée" : `${delta > 0 ? "+" : ""}${delta}% vs estimation ML`;

  return (
    <aside className="predictionPanel" aria-live="polite">
      <div className="verdictHero">
        <span>{mode === "listing" ? "Verdict annonce" : "Avis de valeur"}</span>
        <strong>{prediction.verdict}</strong>
        <small>{deltaLabel}</small>
      </div>

      <div className="priceDecision">
        <div>
          <span>Prix estimé</span>
          <strong>{formatEuro(prediction.predicted_price)}</strong>
          <small>{formatEuro(prediction.predicted_price_m2)}/m²</small>
        </div>
        <div>
          <span>Fourchette</span>
          <strong>{formatEuro(prediction.range_low)} - {formatEuro(prediction.range_high)}</strong>
          <small>Confiance {prediction.confidence.toLowerCase()}</small>
        </div>
      </div>

      <div className="confidenceStrip">
        <strong>Confiance {prediction.confidence}</strong>
        <span>{prediction.confidence_note}</span>
      </div>

      <details className="whyBox" open>
        <summary>Pourquoi ce verdict ?</summary>
        <div className="factorList">
          {prediction.factors.map((factor) => (
            <div className="factorRow" key={factor.label}>
              <span>{factor.label}</span>
              <div><i style={{ width: `${Math.max(factor.importance * 100, 8)}%` }} /></div>
              <strong>{formatNumber(factor.importance * 100, 0)}%</strong>
            </div>
          ))}
        </div>
        <button className="textButton" type="button" onClick={onShowModel}>Voir la transparence du modèle</button>
      </details>
    </aside>
  );
}

function ListingJourney({
  commune,
  query,
  suggestions,
  showSuggestions,
  suggestionLoading,
  parsedListing,
  prediction,
  loading,
  modelReady,
  error,
  onQueryChange,
  onSelectCommune,
  onFocusSearch,
  onBlurSearch,
  onSubmit,
  onShowModel,
}: {
  commune: CommuneDetail | null;
  query: string;
  suggestions: CommuneSummary[];
  showSuggestions: boolean;
  suggestionLoading: boolean;
  parsedListing: ListingParseResult | null;
  prediction: ModelPredictionResult | null;
  loading: boolean;
  modelReady: boolean;
  error: string | null;
  onQueryChange: (value: string) => void;
  onSelectCommune: (code: string) => void;
  onFocusSearch: () => void;
  onBlurSearch: () => void;
  onSubmit: (payload: { url: string; price: number; surface: number }) => void;
  onShowModel: () => void;
}) {
  const [url, setUrl] = useState("");
  const [price, setPrice] = useState("185000");
  const [surface, setSurface] = useState("72");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit({ url, price: Number(price), surface: Number(surface) });
  }

  return (
    <section className="workflowGrid">
      <div className="workflowPanel">
        <div className="sectionTitle productSectionTitle">
          <h2>Évaluer une annonce</h2>
          <span>Parcours particulier</span>
        </div>
        <p className="workflowIntro">
          L’objectif est volontairement court : saisir une annonce, comprendre si le prix est cohérent, puis voir les raisons principales.
        </p>

        <form className="decisionForm" onSubmit={submit}>
          <label className="fullField">
            <span>Lien d’annonce, optionnel</span>
            <input value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://www.seloger.com/..." />
          </label>

          <label>
            <span>Prix annoncé</span>
            <input value={price} onChange={(event) => setPrice(event.target.value)} inputMode="numeric" />
          </label>
          <label>
            <span>Surface</span>
            <input value={surface} onChange={(event) => setSurface(event.target.value)} inputMode="decimal" />
          </label>
          <div className="fullField">
            <span className="fieldLabel">Commune du bien</span>
            <SearchBox
              query={query}
              suggestions={suggestions}
              showSuggestions={showSuggestions}
              loading={suggestionLoading}
              placeholder="Bourg-en-Bresse, Farges..."
              onQueryChange={onQueryChange}
              onSelect={onSelectCommune}
              onFocus={onFocusSearch}
              onBlur={onBlurSearch}
            />
          </div>

          <button className="primaryAction" type="submit" disabled={loading || !commune || !modelReady}>
            {!modelReady ? "Modèle en entraînement..." : loading ? "Analyse ML..." : "Analyser l’annonce"}
          </button>
        </form>

        {parsedListing ? (
          <div className="sourceNotice">
            <strong>{parsedListing.source ?? "Annonce détectée"}</strong>
            <span>{parsedListing.commune ?? "Commune à confirmer"} · prix et surface à saisir manuellement</span>
          </div>
        ) : null}
        {error ? <p className="errorBanner">{error}</p> : null}
      </div>

      <PredictionResult prediction={prediction} mode="listing" onShowModel={onShowModel} />
    </section>
  );
}

function AgentJourney({
  commune,
  query,
  suggestions,
  showSuggestions,
  suggestionLoading,
  nearby,
  prediction,
  loading,
  modelReady,
  error,
  onQueryChange,
  onSelectCommune,
  onFocusSearch,
  onBlurSearch,
  onSubmit,
  onShowModel,
}: {
  commune: CommuneDetail | null;
  query: string;
  suggestions: CommuneSummary[];
  showSuggestions: boolean;
  suggestionLoading: boolean;
  nearby: NearbyCommune[];
  prediction: ModelPredictionResult | null;
  loading: boolean;
  modelReady: boolean;
  error: string | null;
  onQueryChange: (value: string) => void;
  onSelectCommune: (code: string) => void;
  onFocusSearch: () => void;
  onBlurSearch: () => void;
  onSubmit: (payload: { price?: number; surface: number }) => void;
  onShowModel: () => void;
}) {
  const [surface, setSurface] = useState("95");
  const [price, setPrice] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit({ price: price ? Number(price) : undefined, surface: Number(surface) });
  }

  return (
    <section className="workflowStack">
      <div className="workflowGrid">
        <div className="workflowPanel proPanel">
          <div className="sectionTitle productSectionTitle">
            <h2>Estimation professionnelle</h2>
            <span>Parcours agent immobilier</span>
          </div>
          <p className="workflowIntro">
            L’agent obtient une fourchette défendable et des arguments de marché pour préparer un rendez-vous ou challenger un prix vendeur.
          </p>

          <form className="decisionForm" onSubmit={submit}>
            <div className="fullField">
              <span className="fieldLabel">Commune à estimer</span>
              <SearchBox
                query={query}
                suggestions={suggestions}
                showSuggestions={showSuggestions}
                loading={suggestionLoading}
                placeholder="Commune du département 01"
                onQueryChange={onQueryChange}
                onSelect={onSelectCommune}
                onFocus={onFocusSearch}
                onBlur={onBlurSearch}
              />
            </div>
            <label>
              <span>Surface</span>
              <input value={surface} onChange={(event) => setSurface(event.target.value)} inputMode="decimal" />
            </label>
            <label>
              <span>Prix vendeur, optionnel</span>
              <input value={price} onChange={(event) => setPrice(event.target.value)} inputMode="numeric" placeholder="Ex. 265000" />
            </label>
            <button className="primaryAction" type="submit" disabled={loading || !commune || !modelReady}>
              {!modelReady ? "Modèle en entraînement..." : loading ? "Estimation..." : "Générer l’avis de valeur"}
            </button>
          </form>
          {error ? <p className="errorBanner">{error}</p> : null}
        </div>

        <PredictionResult prediction={prediction} mode="agent" onShowModel={onShowModel} />
      </div>

      <ComparisonTable commune={commune} nearby={nearby} compact />
    </section>
  );
}

function FranceMap({
  points,
  metric,
  setMetric,
  selectedCode,
  onSelect,
}: {
  points: MapPoint[];
  metric: MapMetric;
  setMetric: (metric: MapMetric) => void;
  selectedCode?: string;
  onSelect: (codeInsee: string) => void;
}) {
  const legends: Record<MapMetric, string[]> = {
    prix_m2_median: ["+ de 3 500", "2 500 - 3 500", "1 800 - 2 500", "1 300 - 1 800", "- de 1 300"],
    apl_score: ["Désert < 2.5", "APL moyen", "Bon accès"],
    opportunity: ["Signal fort", "Signal moyen", "Signal faible"],
  };

  return (
    <section className="mapCard" aria-label="Carte des communes">
      <div className="mapToggles" aria-label="Indicateur affiché">
        <button className={metric === "prix_m2_median" ? "selected" : ""} onClick={() => setMetric("prix_m2_median")} type="button">
          Prix/m²
        </button>
        <button className={metric === "apl_score" ? "selected" : ""} onClick={() => setMetric("apl_score")} type="button">
          APL
        </button>
        <button className={metric === "opportunity" ? "selected" : ""} onClick={() => setMetric("opportunity")} type="button">
          Opportunité
        </button>
      </div>

      <div className="mapCanvas">
        {points.length > 0 ? (
          <CommuneMap points={points} selectedCode={selectedCode} metric={metric} onSelect={onSelect} />
        ) : (
          <div className="mapLoading">Carte indisponible : coordonnées absentes ou API en chargement.</div>
        )}

        <div className="legend">
          <strong>{metric === "prix_m2_median" ? "Prix médian €/m²" : metric === "apl_score" ? "Score APL" : "Opportunité"}</strong>
          {legends[metric].map((label, index) => (
            <span key={label}>
              <i className={["red", "orange", "yellow", "pale", "green"][index] ?? "green"} />
              {label}
            </span>
          ))}
          <small>Source : DVF, APL, INSEE</small>
        </div>
      </div>
    </section>
  );
}

function CommunePanel({ commune }: { commune: CommuneDetail | null }) {
  if (!commune) {
    return (
      <aside className="communePanel emptyPanel">
        <h2>Commune en chargement</h2>
        <p>Recherche une commune pour afficher les indicateurs.</p>
      </aside>
    );
  }

  const metrics = [
    { label: "Population", value: formatNumber(commune.population), detail: commune.variation_population_2022_2024_pct !== null ? `${commune.variation_population_2022_2024_pct}%` : "2024" },
    { label: "Densité", value: formatNumber(commune.densite), detail: "hab./km²" },
    { label: "Ventes / an", value: formatNumber(commune.nb_ventes), detail: "DVF" },
    { label: "ZRR", value: commune.is_zrr ? "Oui" : "Non", detail: commune.is_zrr ? "Classée" : "Non classée" },
  ];

  return (
    <aside className="communePanel" aria-label="Résumé commune">
      <div className="communeHeader">
        <h2>{commune.commune}</h2>
        <p>{commune.code_insee}</p>
      </div>

      <div className="priceBlock">
        <span>Prix médian observé</span>
        <strong>{formatNumber(commune.prix_m2_median)} €<small>/m²</small></strong>
        <em>{commune.type_zone ?? "Zone locale"}</em>
      </div>

      <div className="healthStrip">
        <div>
          <span>APL</span>
          <strong>{formatNumber(commune.apl_score, 1)}<small>/5</small></strong>
        </div>
        <div>
          <span>Désert médical</span>
          <strong>{commune.desert_medical ? "Oui" : "Non"}</strong>
        </div>
        <span className="medicalIcon" aria-hidden="true">⌕</span>
      </div>

      <div className="scoreBlock">
        <span>Score d’attractivité</span>
        <strong>{commune.score_attractivite}<small>/100</small></strong>
        <div className="scoreTrack"><span style={{ width: `${commune.score_attractivite}%` }} /></div>
      </div>

      <dl className="metricGrid">
        {metrics.map((metric) => (
          <div key={metric.label}>
            <dt>{metric.label}</dt>
            <dd>{metric.value}</dd>
            <small>{metric.detail}</small>
          </div>
        ))}
      </dl>
    </aside>
  );
}

function ComparisonTable({ commune, nearby, compact = false }: { commune: CommuneDetail | null; nearby: NearbyCommune[]; compact?: boolean }) {
  const rows = commune ? [commune, ...nearby] : nearby;

  return (
    <section className={compact ? "comparison proComparison" : "comparison"}>
      <h2>Comparables locaux</h2>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Commune</th>
              <th>Prix médian €/m²</th>
              <th>Distance</th>
              <th>APL (/5)</th>
              <th>Désert médical</th>
              <th>Score attractivité</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((item, index) => (
              <tr key={item.code_insee} className={index === 0 ? "highlightedRow" : ""}>
                <td><strong>{item.commune}</strong> <span>({item.code_insee})</span></td>
                <td>{formatEuro(item.prix_m2_median)}</td>
                <td>{index === 0 ? "Référence" : `${formatNumber((item as NearbyCommune).distance_km, 1)} km`}</td>
                <td>{formatNumber(item.apl_score, 1)}</td>
                <td className={item.desert_medical ? "negative" : "positive"}>{item.desert_medical ? "Oui" : "Non"}</td>
                <td>
                  <div className="scoreCell">
                    <span><i style={{ width: `${item.score_attractivite}%` }} /></span>
                    <b>{item.score_attractivite}/100</b>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelTransparency({ model, health }: { model: ModelSummary | null; health: HealthStatus | null }) {
  const factors = useMemo(() => {
    const importance = model?.summary?.importance ?? [];
    if (importance.length === 0) return [];
    const max = Math.max(...importance.map((item) => item.importance));
    return importance.slice(0, 8).map((item) => ({
      label: item.variable,
      value: Math.round((item.importance / max) * 100),
      raw: Math.round(item.importance * 100),
    }));
  }, [model]);

  return (
    <section className="modelStory">
      <div className="modelHeadline">
        <h2>Transparence du modèle</h2>
        <p>
          Le modèle prédit un prix au m² à partir des signaux communaux disponibles. La V1 est une démonstration sur les
          données DVF chargées pour le département 01.
        </p>
      </div>

      <div className="modelKpis">
        <div>
          <span>Modèle retenu</span>
          <strong>{model?.summary?.nom_modele ?? (model?.loading ? "En entraînement" : "Indisponible")}</strong>
        </div>
        <div>
          <span>Erreur moyenne</span>
          <strong>{formatEuro(model?.summary?.mae)}<small>/m²</small></strong>
        </div>
        <div>
          <span>R² test</span>
          <strong>{model?.summary?.r2 ?? "n/a"}</strong>
        </div>
        <div>
          <span>Communes ML</span>
          <strong>{formatNumber(model?.summary?.n ?? health?.rows)}</strong>
        </div>
      </div>

      <div className="bottomGrid modelBottom">
        <div className="modelPanel">
          <h2>Variables qui pèsent le plus</h2>
          <div className="factorList">
            {(factors.length ? factors : [{ label: "Modèle en cours d’entraînement", value: 35, raw: 0 }]).map((factor) => (
              <div className="factorRow" key={factor.label}>
                <span>{factor.label}</span>
                <div><i style={{ width: `${factor.value}%` }} /></div>
                <strong>{Number.isFinite(factor.raw) ? factor.raw : "..." }%</strong>
              </div>
            ))}
          </div>
        </div>

        <aside className="takeawayPanel">
          <h2>Limites assumées</h2>
          <p><span className="greenDot" aria-hidden="true" />Prototype entraîné sur les données locales disponibles, actuellement département 01.</p>
          <p><span className="amberDot" aria-hidden="true" />L’estimation est indicative : elle ne remplace pas une visite, un mandat ou une expertise réglementée.</p>
          <p><span className="greenDot" aria-hidden="true" />La confiance dépend du nombre de ventes et des variables disponibles pour la commune.</p>
        </aside>
      </div>
    </section>
  );
}

export default function Home() {
  const [activeView, setActiveView] = useState<View>("home");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<CommuneSummary[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestionLoading, setSuggestionLoading] = useState(false);
  const [commune, setCommune] = useState<CommuneDetail | null>(null);
  const [nearby, setNearby] = useState<NearbyCommune[]>([]);
  const [model, setModel] = useState<ModelSummary | null>(null);
  const [points, setPoints] = useState<MapPoint[]>([]);
  const [metric, setMetric] = useState<MapMetric>("prix_m2_median");
  const [listingPrediction, setListingPrediction] = useState<ModelPredictionResult | null>(null);
  const [agentPrediction, setAgentPrediction] = useState<ModelPredictionResult | null>(null);
  const [parsedListing, setParsedListing] = useState<ListingParseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCommune = useCallback(async (codeInsee: string) => {
    setLoading(true);
    setError(null);
    try {
      const [detail, close] = await Promise.all([api.commune(codeInsee), api.nearby(codeInsee)]);
      setCommune(detail);
      setNearby(close.items);
      setQuery(detail.commune);
      setSuggestions([]);
      setShowSuggestions(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }, []);

  const updateQuery = useCallback((value: string) => {
    setQuery(value);
    setShowSuggestions(true);
    if (value.trim().length === 0) {
      setSuggestions([]);
      setSuggestionLoading(false);
    } else {
      setSuggestionLoading(true);
    }
  }, []);

  useEffect(() => {
    const value = query.trim();
    if (!showSuggestions || value.length === 0) return;

    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const result = await api.searchCommunes(value);
        if (!cancelled) setSuggestions(result.items);
      } catch {
        if (!cancelled) setSuggestions([]);
      } finally {
        if (!cancelled) setSuggestionLoading(false);
      }
    }, 220);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, showSuggestions]);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      try {
        const status = await api.health();
        if (cancelled) return;
        setHealth(status);
        const [mapResult, modelResult] = await Promise.allSettled([api.map("prix_m2_median"), api.modelSummary()]);
        if (cancelled) return;
        if (mapResult.status === "fulfilled") setPoints(mapResult.value.items);
        if (modelResult.status === "fulfilled") setModel(modelResult.value);
        if (status.dataset_ready) {
          const result = await api.searchCommunes("");
          if (!cancelled && result.items[0]) await loadCommune(result.items[0].code_insee);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "API indisponible");
      }
    }
    boot();
    return () => {
      cancelled = true;
    };
  }, [loadCommune]);

  useEffect(() => {
    api.map(metric).then((result) => setPoints(result.items)).catch(() => setPoints([]));
  }, [metric]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      api.health().then(setHealth).catch(() => undefined);
      api.modelSummary().then(setModel).catch(() => undefined);
    }, 8000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [activeView]);

  async function selectCommune(code: string) {
    await loadCommune(code);
  }

  async function submitListing(payload: { url: string; price: number; surface: number }) {
    if (!payload.price || !payload.surface) {
      setError("Renseigne au minimum un prix annoncé et une surface.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      let codeInsee = commune?.code_insee;
      if (payload.url.trim() && looksLikeUrl(payload.url)) {
        const parsed = await api.parseListing(payload.url.trim());
        setParsedListing(parsed);
        if (parsed.code_insee) {
          codeInsee = parsed.code_insee;
          await loadCommune(parsed.code_insee);
        } else if (parsed.commune) {
          const result = await api.searchCommunes(parsed.commune);
          if (result.items[0]) {
            codeInsee = result.items[0].code_insee;
            await loadCommune(result.items[0].code_insee);
          }
        }
        if (!parsed.supported) setError("Lien détecté, mais portail non supporté pour le moment.");
      }

      if (!codeInsee && !query.trim()) throw new Error("Sélectionne une commune du département 01.");
      const result = await api.predictModel({
        code_insee: codeInsee,
        commune: codeInsee ? undefined : query.trim(),
        price: payload.price,
        surface: payload.surface,
      });
      setListingPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prédiction indisponible");
    } finally {
      setLoading(false);
    }
  }

  async function submitAgent(payload: { price?: number; surface: number }) {
    if (!payload.surface) {
      setError("Renseigne une surface.");
      return;
    }
    if (!commune && !query.trim()) {
      setError("Sélectionne une commune du département 01.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await api.predictModel({
        code_insee: commune?.code_insee,
        commune: commune ? undefined : query.trim(),
        price: payload.price,
        surface: payload.surface,
      });
      setAgentPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Estimation indisponible");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <TopNav activeView={activeView} health={health} onNavigate={setActiveView} />

      {activeView === "home" ? <HomeHero onNavigate={setActiveView} /> : null}

      {activeView === "search" ? (
        <SmartSearch
          query={query}
          suggestions={suggestions}
          showSuggestions={showSuggestions}
          suggestionLoading={suggestionLoading}
          selectedCommune={commune}
          onQueryChange={updateQuery}
          onSelectCommune={selectCommune}
          onFocusSearch={() => {
            setShowSuggestions(true);
            if (query.trim()) setSuggestionLoading(true);
          }}
          onBlurSearch={() => window.setTimeout(() => setShowSuggestions(false), 120)}
          onSelectResult={async (code) => {
            await loadCommune(code);
            setActiveView("market");
          }}
        />
      ) : null}

      {activeView === "listing" ? (
        <ListingJourney
          commune={commune}
          query={query}
          suggestions={suggestions}
          showSuggestions={showSuggestions}
          suggestionLoading={suggestionLoading}
          parsedListing={parsedListing}
          prediction={listingPrediction}
          loading={loading}
          modelReady={Boolean(health?.model_ready)}
          error={error}
          onQueryChange={updateQuery}
          onSelectCommune={selectCommune}
          onFocusSearch={() => {
            setShowSuggestions(true);
            if (query.trim()) setSuggestionLoading(true);
          }}
          onBlurSearch={() => window.setTimeout(() => setShowSuggestions(false), 120)}
          onSubmit={submitListing}
          onShowModel={() => setActiveView("model")}
        />
      ) : null}

      {activeView === "agent" ? (
        <AgentJourney
          commune={commune}
          query={query}
          suggestions={suggestions}
          showSuggestions={showSuggestions}
          suggestionLoading={suggestionLoading}
          nearby={nearby}
          prediction={agentPrediction}
          loading={loading}
          modelReady={Boolean(health?.model_ready)}
          error={error}
          onQueryChange={updateQuery}
          onSelectCommune={selectCommune}
          onFocusSearch={() => {
            setShowSuggestions(true);
            if (query.trim()) setSuggestionLoading(true);
          }}
          onBlurSearch={() => window.setTimeout(() => setShowSuggestions(false), 120)}
          onSubmit={submitAgent}
          onShowModel={() => setActiveView("model")}
        />
      ) : null}

      {activeView === "market" ? (
        <>
          <section className="marketIntro">
            <h1>Marché local</h1>
            <p>La carte et les indicateurs restent disponibles, mais comme preuves de contexte autour de l’estimation ML.</p>
          </section>
          <div className="analysisGrid">
            <FranceMap points={points} metric={metric} setMetric={setMetric} selectedCode={commune?.code_insee} onSelect={loadCommune} />
            <CommunePanel commune={commune} />
          </div>
          <ComparisonTable commune={commune} nearby={nearby} />
        </>
      ) : null}

      {activeView === "model" ? <ModelTransparency model={model} health={health} /> : null}

      <footer className="footer">
        <span>Données : DVF, INSEE, APL, OFGL, ZRR · Couverture prototype : département 01.</span>
        <span>Territoire Immo fournit une estimation indicative, pas un diagnostic financier ou réglementaire.</span>
      </footer>
    </main>
  );
}
