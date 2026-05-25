"use client";

import { FormEvent, useState } from "react";
import {
  api,
  type CommuneDetail,
  type CommuneSummary,
  type SmartSearchResponse,
  type SmartSearchZone,
} from "@/lib/api";

type SmartSearchProps = {
  query: string;
  suggestions: CommuneSummary[];
  showSuggestions: boolean;
  suggestionLoading: boolean;
  selectedCommune: CommuneDetail | null;
  onQueryChange: (value: string) => void;
  onSelectCommune: (code: string) => void;
  onFocusSearch: () => void;
  onBlurSearch: () => void;
  onSelectResult: (code: string) => void;
};

const zoneTypes: SmartSearchZone[] = ["Tous", "Rural", "Périurbain", "Urbain"];

function formatNumber(value?: number | null, maximumFractionDigits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits }).format(value);
}

function formatEuro(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${formatNumber(value)} €`;
}

function deltaLabel(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return value >= 0 ? `${formatEuro(value)} restant` : `${formatEuro(Math.abs(value))} au-dessus`;
}

export default function SmartSearch({
  query,
  suggestions,
  showSuggestions,
  suggestionLoading,
  selectedCommune,
  onQueryChange,
  onSelectCommune,
  onFocusSearch,
  onBlurSearch,
  onSelectResult,
}: SmartSearchProps) {
  const [budget, setBudget] = useState("220000");
  const [surface, setSurface] = useState("80");
  const [radiusKm, setRadiusKm] = useState("40");
  const [minApl, setMinApl] = useState("2.5");
  const [zoneType, setZoneType] = useState<SmartSearchZone>("Tous");
  const [requireBudgetFit, setRequireBudgetFit] = useState(true);
  const [strictApl, setStrictApl] = useState(false);
  const [results, setResults] = useState<SmartSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const shouldShowDropdown = showSuggestions && (suggestions.length > 0 || suggestionLoading || query.trim().length > 0);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const parsedBudget = Number(budget);
    const parsedSurface = Number(surface);
    if (!parsedBudget || !parsedSurface || parsedBudget <= 0 || parsedSurface <= 0) {
      setError("Renseigne un budget et une surface valides.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await api.smartSearch({
        budget: parsedBudget,
        surface: parsedSurface,
        pivot_code_insee: selectedCommune?.code_insee ?? null,
        radius_km: radiusKm === "all" ? null : Number(radiusKm),
        min_apl: minApl ? Number(minApl) : null,
        zone_type: zoneType,
        require_budget_fit: requireBudgetFit,
        strict_apl: strictApl,
        limit: 20,
      });
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recherche indisponible");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="smartSearchPage">
      <div className="smartSearchHeader">
        <span>Recherche intelligente</span>
        <h1>Trouvez les communes compatibles avec votre projet.</h1>
        <p>Classement selon budget, surface, accès aux soins et dynamique locale.</p>
      </div>

      <div className="smartSearchLayout">
        <form className="smartSearchPanel" onSubmit={submit}>
          <label>
            <span>Budget total</span>
            <input value={budget} min="50000" max="1000000" step="10000" inputMode="numeric" onChange={(event) => setBudget(event.target.value)} />
          </label>
          <label>
            <span>Surface souhaitée</span>
            <input value={surface} min="20" max="300" step="5" inputMode="numeric" onChange={(event) => setSurface(event.target.value)} />
          </label>
          <label className="fullField">
            <span>Commune pivot</span>
            <div className="searchField compactSearch">
              <span className="searchIcon" aria-hidden="true" />
              <input
                aria-label="Commune pivot"
                autoComplete="off"
                placeholder="Bourg-en-Bresse, Oyonnax..."
                value={query}
                onBlur={onBlurSearch}
                onChange={(event) => onQueryChange(event.target.value)}
                onFocus={onFocusSearch}
              />
              {shouldShowDropdown ? (
                <div className="suggestions" role="listbox" aria-label="Communes trouvées">
                  {suggestions.map((item) => (
                    <button
                      type="button"
                      key={item.code_insee}
                      onClick={() => onSelectCommune(item.code_insee)}
                      onMouseDown={(event) => event.preventDefault()}
                    >
                      <strong>{item.commune}</strong>
                      <span>{item.code_insee}</span>
                      <small>{formatEuro(item.prix_m2_median)}/m²</small>
                    </button>
                  ))}
                  {suggestionLoading ? <div className="suggestionState">Recherche...</div> : null}
                  {!suggestionLoading && suggestions.length === 0 && query.trim() ? <div className="suggestionState">Aucune commune trouvée</div> : null}
                </div>
              ) : null}
            </div>
          </label>
          <label>
            <span>Rayon</span>
            <select value={radiusKm} onChange={(event) => setRadiusKm(event.target.value)}>
              <option value="10">10 km</option>
              <option value="20">20 km</option>
              <option value="40">40 km</option>
              <option value="60">60 km</option>
              <option value="100">100 km</option>
              <option value="all">Tout le département</option>
            </select>
          </label>
          <label>
            <span>APL minimum</span>
            <input value={minApl} min="0" max="5" step="0.1" inputMode="decimal" onChange={(event) => setMinApl(event.target.value)} />
          </label>
          <label>
            <span>Type de zone</span>
            <select value={zoneType} onChange={(event) => setZoneType(event.target.value as SmartSearchZone)}>
              {zoneTypes.map((zone) => <option key={zone} value={zone}>{zone}</option>)}
            </select>
          </label>
          <label className="checkField fullField">
            <input type="checkbox" checked={requireBudgetFit} onChange={(event) => setRequireBudgetFit(event.target.checked)} />
            <span>Afficher seulement les communes compatibles avec le budget</span>
          </label>
          <label className="checkField fullField">
            <input type="checkbox" checked={strictApl} onChange={(event) => setStrictApl(event.target.checked)} />
            <span>Exclure les communes sous le seuil APL</span>
          </label>
          <button className="primaryAction" type="submit" disabled={loading}>
            {loading ? "Recherche..." : "Classer les communes"}
          </button>
          {error ? <p className="errorBanner fullField">{error}</p> : null}
        </form>

        <section className="smartSearchResults" aria-live="polite">
          {results ? (
            <>
              <div className="searchSummary">
                <strong>{results.summary.total_matches} communes trouvées</strong>
                <span>Budget : {formatEuro(results.summary.budget)}</span>
                <span>Surface : {formatNumber(results.summary.surface)} m²</span>
                <span>Prix cible max : {formatEuro(results.summary.max_price_m2)}/m²</span>
                <span>{results.summary.require_budget_fit ? "Budget compatible uniquement" : "Budget indicatif"}</span>
                {results.summary.radius_ignored ? <em>Rayon ignoré : commune pivot ou coordonnées indisponibles.</em> : null}
              </div>

              {results.items.length === 0 ? (
                <div className="emptySearchState">
                  Aucune commune du dataset ne permet cette surface avec ce budget. Augmentez le budget, réduisez la surface ou désactivez le filtre budget compatible.
                </div>
              ) : (
                <div className="resultList">
                  {results.items.map((item) => (
                    <article className="communeResultCard" key={item.code_insee}>
                      <div className="resultRank">#{item.rank}</div>
                      <div className="resultMain">
                        <h2>{item.commune}</h2>
                        <p>{item.reason}</p>
                      </div>
                      <strong className="resultScore">{item.score}/100</strong>
                      <dl className="resultMetrics">
                        <div><dt>Prix estimé</dt><dd>{formatEuro(item.estimated_price)}</dd></div>
                        <div><dt>Budget</dt><dd>{deltaLabel(item.budget_delta)}</dd></div>
                        <div><dt>APL</dt><dd>{formatNumber(item.apl_score, 1)} / 5</dd></div>
                        <div><dt>Dynamique</dt><dd>{formatNumber(item.variation_population_2022_2024_pct, 1)} %</dd></div>
                        <div><dt>Ventes</dt><dd>{formatNumber(item.nb_ventes)}</dd></div>
                        <div><dt>Distance</dt><dd>{item.distance_km === null ? "Département" : `${formatNumber(item.distance_km, 1)} km`}</dd></div>
                      </dl>
                      <div className="resultBadges">
                        {item.badges.map((badge) => <span key={badge}>{badge}</span>)}
                      </div>
                      <button className="textButton resultMapButton" type="button" onClick={() => onSelectResult(item.code_insee)}>
                        Voir sur la carte
                      </button>
                    </article>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="smartSearchEmpty">
              <strong>Le classement apparaîtra ici.</strong>
              <p>Score indicatif construit à partir des données disponibles sur le département 01.</p>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
