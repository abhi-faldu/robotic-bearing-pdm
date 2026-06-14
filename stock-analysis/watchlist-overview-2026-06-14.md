# Watchlist Universe Analysis — Overview Report

**Prepared:** 2026-06-14 · **Names covered:** 52 · **Format:** Equity research / IB-style screening overview

> **DRAFT WORK PRODUCT FOR HUMAN REVIEW — NOT INVESTMENT ADVICE.** This report does
> not make buy/sell recommendations, does not execute transactions, and does not
> constitute legal, tax, or accounting advice. Methodology built on the Anthropic
> *Claude for Financial Services* `sector-overview` / `competitive-analysis`
> skills. Past performance is not a reliable indicator of future results.

---

## 1. Methodology & Data Provenance

| Data element | Source | Confidence |
|---|---|---|
| Price & day % change | User watchlist screenshots (snapshot, ~Jun 2026) | As-observed |
| Business / product descriptions | Model knowledge (reliable through early 2026) | High |
| Thematic & competitive positioning | Model knowledge | High |
| Latest-quarter earnings figures | **Deferred to deep-dive (live web research)** | Verify before use |
| Market-cap buckets | Approximate, derived from snapshot price | Indicative |

**Important:** Paid data connectors (FactSet, Daloopa, Morningstar, S&P Capital IQ,
etc.) are *not* configured in this environment. Anywhere this report would normally
cite a hard, current financial metric, it instead flags `[verify]`. The deep-dive
stage pulls those numbers live.

A few snapshot prices are notably high versus early-2026 levels (e.g., MU $989.60,
GEV $942.60, LITE ~$921, TDG $1,256). These are reproduced **exactly as displayed**
in the watchlist; they are not adjusted or second-guessed here.

---

## 2. Universe Snapshot (sorted by day % change)

### Decliners
| Ticker | Company | Price | Day % | Theme |
|---|---|---|---|---|
| SIDU | Sidus Space | $3.75 | ▼14.58% | Space |
| LUNR | Intuitive Machines | $26.79 | ▼12.57% | Space |
| RDW | Redwire | $15.20 | ▼11.06% | Space |
| SATL | Satellogic | $6.72 | ▼10.64% | Space |
| SATS | EchoStar | $114.71 | ▼10.47% | Space / Telecom |
| LPTH | LightPath Technologies | $14.57 | ▼6.84% | Photonics / Defense |
| HIMS | Hims & Hers Health | $26.91 | ▼6.79% | Healthcare |
| SPIR | Spire Global | $18.50 | ▼6.52% | Space |
| RDDT | Reddit | $163.01 | ▼5.92% | Consumer Internet |
| IRDM | Iridium Communications | $47.29 | ▼5.25% | Space / Telecom |
| KTOS | Kratos Defense | $58.23 | ▼0.94% | Defense |
| AVGO | Broadcom | $382.55 | ▼0.78% | AI Semis |
| BB | BlackBerry | $9.22 | ▼0.75% | Software / IoT |
| CSCO | Cisco | $120.97 | ▼0.71% | Networking |
| MU | Micron Technology | $989.60 | ▼0.63% | AI Semis (memory) |
| SOFI | SoFi Technologies | $16.58 | ▼0.54% | Fintech |
| SOUN | SoundHound AI | $6.97 | ▼0.50% | AI Software |
| CRM | Salesforce | $165.70 | ▼0.45% | Enterprise SW |
| ADTN | Adtran | $15.15 | ▼0.33% | Networking |
| ISRG | Intuitive Surgical | $411.71 | ▼0.29% | MedTech |
| SNPS | Synopsys | $455.20 | ▼0.24% | EDA / Semis |
| NOW | ServiceNow | $102.88 | ▼0.19% | Enterprise SW |
| STM | STMicroelectronics | $78.01 | ▼0.14% | Semis |
| TDG | TransDigm | $1,256.05 | ▼0.12% | Aerospace |
| OKLO | Oklo | $57.85 | ▼0.02% | Nuclear |

### Gainers
| Ticker | Company | Price | Day % | Theme |
|---|---|---|---|---|
| MSFT | Microsoft | $390.68 | ▲0.09% | Enterprise SW / Cloud |
| BAH | Booz Allen Hamilton | $77.25 | ▲0.06% | Defense / Consulting |
| TTWO | Take-Two Interactive | $212.20 | ▲0.06% | Gaming |
| SPGI | S&P Global | $414.12 | ▲0.19% | Financial Data |
| MRVL | Marvell Technology | $281.20 | ▲0.18% | AI Semis |
| BWXT | BWX Technologies | $194.96 | ▲0.14% | Nuclear / Defense |
| LUMN | Lumen Technologies | $8.51 | ▲0.23% | Networking |
| AVR | Anteris Technologies | $9.33 | ▲0.22% | MedTech |
| NVDA | NVIDIA | $205.42 | ▲0.27% | AI Semis |
| EXI2 | iShares DJ Global Titans 50 ETF | €108.53 | ▲0.29% | ETF (mega-cap) |
| ETN | Eaton | $395.00 | ▲0.34% | Electrification |
| IONQ | IonQ | $58.26 | ▲0.47% | Quantum |
| POWI | Power Integrations | $78.15 | ▲0.50% | Semis (power) |
| SAP | SAP | $164.93 | ▲0.79% | Enterprise SW |
| S | SentinelOne | $14.88 | ▲0.80% | Cybersecurity |
| MARA | Marathon Digital | $14.10 | ▲3.60% | Crypto Mining |
| IOT | Samsara | $33.44 | ▲3.67% | IoT / AI Software |
| PWR | Quanta Services | $709.00 | ▲3.76% | Power Infrastructure |
| GEV | GE Vernova | $942.60 | ▲3.95% | Power / Electrification |
| ABTC | American Bitcoin Corp | $0.87 | ▲4.38% | Crypto Mining |
| SMR | NuScale Power | $10.01 | ▲4.56% | Nuclear |
| ANET | Arista Networks | $163.76 | ▲4.71% | AI Networking |
| QCOM | Qualcomm | $212.65 | ▲4.77% | Semis |
| ICHR | Ichor Holdings | $88.08 | ▲4.81% | Semicap |
| SARO | StandardAero | $27.20 | ▲4.82% | Aerospace (MRO) |
| PATH | UiPath | $10.54 | n/a | AI Software (RPA) |
| LITE | Lumentum Holdings | ~$921.00 | (partial) | AI Optics |

---

## 3. Portfolio-Level Read

**This is a high-beta, theme-driven growth watchlist** clustered around five
secular narratives: **(1) AI compute & infrastructure**, **(2) the power/nuclear
build-out to feed AI data centers**, **(3) space & satellites**, **(4) defense
modernization**, and **(5) crypto**. It is deliberately tilted toward small/mid-cap
"picks-and-shovels" beneficiaries rather than defensive compounders.

**What the day's tape is telling you:**
- **Speculative space names were dumped hard** — SIDU −14.6%, LUNR −12.6%,
  RDW −11.1%, SATL −10.6%, SATS −10.5%, SPIR −6.5%, IRDM −5.3%. When a whole
  sub-theme moves down together with that magnitude, it is usually a *factor/risk-off*
  move (rates, capital-markets jitters, a sector-specific headline, or a funding/
  dilution scare) rather than 7 separate company events. **`[verify catalyst]`**
- **Power & nuclear and AI-networking led the upside** — GEV +3.95%, SMR +4.56%,
  QCOM +4.77%, ANET +4.71%, ICHR +4.81%, SARO +4.82%, PWR +3.76%. The "electrons
  for AI" and "AI back-end network" trades were in favor.
- **Mega-cap AI/software was flat-to-soft** — NVDA +0.27%, MSFT +0.09%,
  AVGO −0.78%, CRM −0.45%, NOW −0.19%. Quiet, low-dispersion day at the top.

**Concentration / risk observations:**
- Heavy single-factor exposure: a large fraction of names rise and fall together
  on the **AI-capex** and **rates** factors. Diversification is lower than the
  52-name count suggests.
- **Profitability spectrum is wide:** from cash-generative mega-caps (MSFT, NVDA,
  AVGO, SPGI, ETN, TDG) to pre-/early-revenue speculatives (SIDU, SATL, LUNR, RDW,
  OKLO, SMR, ABTC, SOUN). Position sizing should reflect that.
- **Two crypto-mining proxies** (MARA, ABTC) add direct bitcoin-price beta.
- **One ETF** (EXI2, iShares DJ Global Titans 50) is the only diversified holding.

---

## 4. Thematic Deep Sections

### 4.1 AI Semiconductors & Compute
*NVDA, AVGO, MRVL, MU, QCOM, STM, POWI, ICHR, SNPS, LITE*

The core "AI infrastructure" sleeve.
- **NVDA — NVIDIA ($205.42):** Dominant accelerator/GPU franchise + CUDA software
  moat; the single biggest beneficiary of AI training/inference capex. Risk: customer
  concentration, China export rules, eventual capex digestion. `[verify latest qtr]`
- **AVGO — Broadcom ($382.55, ▼0.78%):** Custom AI ASICs (hyperscaler accelerators)
  + networking silicon + VMware software. A primary "merchant + custom" AI play.
  Risk: lumpy ASIC ramps, large debt from VMware.
- **MRVL — Marvell ($281.20, ▲0.18%):** Custom silicon + optical DSPs + datacenter
  networking. Leveraged to AI back-end connectivity. Risk: program timing, competition.
- **MU — Micron ($989.60, ▼0.63%):** Memory (DRAM/NAND) and **HBM** — the memory
  attached to every AI accelerator. Highly cyclical but HBM has tightened the cycle.
  *Snapshot price is far above early-2026 levels — `[verify price/possible action]`.*
- **QCOM — Qualcomm ($212.65, ▲4.77%):** Mobile/edge SoCs diversifying into
  auto, IoT, PC, and on-device AI. Day's strong move `[verify catalyst]`. Risk:
  Apple modem in-sourcing, handset cyclicality.
- **STM — STMicroelectronics ($78.01, ▼0.14%):** Broad analog/MCU/power; auto &
  industrial cyclical exposure. Risk: industrial/auto inventory cycle.
- **POWI — Power Integrations ($78.15, ▲0.50%):** High-voltage power conversion ICs;
  GaN; benefits from electrification & fast-charging. Risk: consumer/industrial demand.
- **ICHR — Ichor Holdings ($88.08, ▲4.81%):** Fluid-delivery subsystems for
  semicap (WFE) — a leveraged play on equipment spending. Strong day `[verify]`.
- **SNPS — Synopsys ($455.20, ▼0.24%):** EDA duopoly (with Cadence) + IP; structural
  toll-booth on all chip design; Ansys deal expands into simulation. Risk: deal
  integration, semis R&D budgets.
- **LITE — Lumentum (~$921, partial):** Optical components/transceivers & lasers for
  AI datacenter interconnect. *Snapshot price far above early-2026 — `[verify]`.*

### 4.2 AI Networking & Connectivity
*ANET, CSCO, ADTN, LUMN, BB*

- **ANET — Arista ($163.76, ▲4.71%):** High-speed datacenter switching (Ethernet for
  AI back-end), software-driven (EOS). Top-tier AI-network beneficiary. Risk: hyperscaler
  concentration, white-box competition.
- **CSCO — Cisco ($120.97, ▼0.71%):** Enterprise networking + security + Splunk
  observability; AI-infra orders building. Mature, cash-generative. Risk: slower growth.
- **ADTN — Adtran ($15.15, ▼0.33%):** Fiber-access & broadband infrastructure
  (BEAD/fiber buildout). Smaller-cap, turnaround-flavored. Risk: margins, balance sheet.
- **LUMN — Lumen ($8.51, ▲0.23%):** Fiber/network operator repositioning around AI
  connectivity ("PCF" deals with hyperscalers); heavy debt. Higher-risk turnaround.
- **BB — BlackBerry ($9.22, ▼0.75%):** QNX automotive OS + cybersecurity (IoT/
  embedded). Story is QNX design-win ramp. Risk: execution, cyber competition.

### 4.3 Enterprise & AI Software
*MSFT, CRM, NOW, SAP, S, PATH, SOUN, IOT, TTWO*

- **MSFT — Microsoft ($390.68, ▲0.09%):** Azure + Copilot/OpenAI; the mega-cap AI
  monetization bellwether. Risk: AI-capex ROI scrutiny, cloud growth deceleration.
- **CRM — Salesforce ($165.70, ▼0.45%):** CRM suite + Agentforce (agentic AI) +
  Data Cloud. Margin-expansion + AI-attach story. Risk: seat-based model vs. agentic
  pricing transition.
- **NOW — ServiceNow ($102.88, ▼0.19%):** Workflow platform; strong AI (Now Assist)
  attach; durable growth. *Snapshot price low vs. early-2026 — `[verify/possible split]`.*
- **SAP — SAP ($164.93, ▲0.79%):** ERP cloud migration (RISE) + Business AI. Steady
  large-cap compounder. Risk: migration pace.
- **S — SentinelOne ($14.88, ▲0.80%):** AI-native endpoint/XDR security; share-gainer
  vs. legacy. Risk: CRWD competition, path to GAAP profit.
- **PATH — UiPath ($10.54):** RPA pivoting to agentic automation. Risk: growth
  deceleration, AI disruption of classic RPA.
- **SOUN — SoundHound AI ($6.97, ▼0.50%):** Conversational/voice AI (auto, QSR,
  IoT). High-multiple, small-revenue speculative. Risk: dilution, customer concentration.
- **IOT — Samsara ($33.44, ▲3.67%):** Connected-operations (fleet/IoT) platform with
  strong ARR growth. Risk: SMB/transport cyclicality, valuation.
- **TTWO — Take-Two ($212.20, ▲0.06%):** Gaming (GTA VI catalyst). Risk: release
  timing, franchise reliance.

### 4.4 Space & Satellites
*IRDM, SPIR, SATS, SATL, RDW, LUNR, SIDU*

The sleeve that **led the day's decline** — likely a sector/factor move `[verify]`.
- **IRDM — Iridium ($47.29, ▼5.25%):** Profitable, cash-generative LEO voice/data
  network (the most mature name here). Risk: D2D-satellite competition, capex cycle.
- **SATS — EchoStar ($114.71, ▼10.47%):** DISH/Boost wireless + Hughes satellite +
  spectrum holdings; spectrum monetization & balance-sheet story. High volatility,
  large move today `[verify catalyst]`.
- **SPIR — Spire Global ($18.50, ▼6.52%):** Space-based data (weather, AIS, RF) +
  "space-as-a-service." Smaller-cap, funding-sensitive.
- **RDW — Redwire ($15.20, ▼11.06%):** Space infrastructure/components + drones
  (Edge Autonomy). M&A-driven. Risk: integration, dilution.
- **LUNR — Intuitive Machines ($26.79, ▼12.57%):** Lunar landers + NASA contracts
  (CLPS, near-space network). Milestone/mission-dependent, lumpy. High risk.
- **SATL — Satellogic ($6.72, ▼10.64%):** Earth-observation/imagery constellation.
  Pre-scale, funding-sensitive speculative.
- **SIDU — Sidus Space ($3.75, ▼14.58%):** Micro-cap space + AI ("FeatherEdge").
  Highly speculative, dilution-prone; biggest decliner today.

### 4.5 Nuclear & Power / Electrification
*OKLO, SMR, GEV, ETN, PWR, BWXT*

The "AI needs electrons" trade — **led the upside today.**
- **GEV — GE Vernova ($942.60, ▲3.95%):** Gas turbines + grid equipment + wind;
  prime beneficiary of data-center power demand & grid capex. Risk: wind drag,
  valuation. *Snapshot price elevated — `[verify]`.*
- **ETN — Eaton ($395.00, ▲0.34%):** Electrical components/power management for
  data centers & electrification. Quality compounder. Risk: cyclicality, valuation.
- **PWR — Quanta Services ($709.00, ▲3.76%):** Electrical/grid infrastructure
  construction; multi-year backlog. Risk: labor, project execution.
- **BWXT — BWX Technologies ($194.96, ▲0.14%):** Naval nuclear reactors (defense) +
  SMR/medical isotopes. Defense-backed + nuclear optionality. Lower-risk nuclear play.
- **OKLO — Oklo ($57.85, ▼0.02%):** Pre-revenue advanced fission (Aurora) developer;
  pure optionality/story stock. Very high risk, pre-commercial.
- **SMR — NuScale ($10.01, ▲4.56%):** SMR developer (NRC-design progress); pre-scale,
  funding/contract-dependent. Very high risk.

### 4.6 Defense & Aerospace
*KTOS, TDG, BAH, SARO, (BWXT — dual)*

- **TDG — TransDigm ($1,256.05, ▼0.12%):** Proprietary aerospace aftermarket parts;
  serial-acquirer, high-margin, levered compounder. Risk: leverage, regulatory pricing scrutiny.
- **KTOS — Kratos ($58.23, ▼0.94%):** Drones, hypersonics, space, propulsion;
  unmanned/missile-defense leverage. Risk: program timing, margins.
- **BAH — Booz Allen ($77.25, ▲0.06%):** Defense/intel consulting + AI for govt
  (Advana). Risk: budget/contracting cycles, DOGE-style efficiency pressure.
- **SARO — StandardAero ($27.20, ▲4.82%):** Aircraft engine MRO; aftermarket-driven,
  post-IPO. Risk: leverage, OEM dependency.

### 4.7 Fintech & Crypto
*SOFI, MARA, ABTC, (RDDT in §4.8)*

- **SOFI — SoFi ($16.58, ▼0.54%):** Digital bank + financial-services platform +
  tech (Galileo). Growth + improving profitability. Risk: credit cycle, rates.
- **MARA — Marathon Digital ($14.10, ▲3.60%):** Bitcoin miner / BTC treasury; direct
  bitcoin-price beta + power costs. High volatility.
- **ABTC — American Bitcoin Corp ($0.87, ▲4.38%):** Sub-$1 micro-cap bitcoin-mining
  proxy; very high risk/speculative.

### 4.8 Healthcare / MedTech, Consumer & Other
*ISRG, HIMS, AVR, RDDT, SPGI, EXI2*

- **ISRG — Intuitive Surgical ($411.71, ▼0.29%):** da Vinci robotic surgery +
  recurring instruments/accessories; durable razor/blade compounder. Risk: valuation,
  procedure-growth deceleration.
- **HIMS — Hims & Hers ($26.91, ▼6.79%):** Telehealth/consumer health subscriptions
  (incl. GLP-1 dynamics). High-growth, headline-sensitive (big down day `[verify]`).
- **AVR — Anteris Technologies ($9.33, ▲0.22%):** Structural-heart (TAVR — DurAVR)
  clinical-stage medtech. Binary trial/regulatory risk; pre-commercial.
- **RDDT — Reddit ($163.01, ▼5.92%):** Social platform; advertising + AI data-licensing.
  High beta to ad cycle & AI-licensing narrative (down sharply today `[verify]`).
- **SPGI — S&P Global ($414.12, ▲0.19%):** Ratings + indices + market data; wide-moat
  compounder; the highest-quality "financial data" name here.
- **EXI2 — iShares DJ Global Titans 50 ETF (€108.53):** Diversified mega-cap global
  equity ETF — the portfolio's lone broad-market diversifier.

---

## 5. Recommended Deep-Dive Shortlist

For the next stage (full **IB-style** + **earnings-review** treatment with **live
web research**), I recommend a balanced 6-name slate spanning the watchlist's main
factors. Adjust freely.

| # | Ticker | Why it earns a deep dive |
|---|---|---|
| 1 | **NVDA** | Anchor of the entire AI-capex thesis; sets the tone for ~15 names here |
| 2 | **GEV** | Cleanest "power-for-AI" leader; led today's up-move; high snapshot price to validate |
| 3 | **MU** | HBM/memory cycle; striking snapshot price warrants a real earnings/valuation check |
| 4 | **LUNR** | Representative high-risk space name; −12.6% day — dissect the catalyst |
| 5 | **SOFI** | Best fintech profitability-inflection story in the list |
| 6 | **ISRG** | Highest-quality compounder; useful "quality anchor" contrast |

---

## 6. Next Steps

1. **Confirm or edit the shortlist** above (or give me your own ~5–8 names).
2. For each chosen name I will run, with live web research:
   - `earnings-analysis` → latest-quarter beat/miss, KPIs, guidance, thesis impact
   - `competitive-analysis` / `comps-analysis` → peer multiples & positioning
   - An **IB-style one-pager** (business, financials, valuation, catalysts, risks)
   - Optional `dcf-model` (Excel) for valuation-sensitive names
3. Deliverables land as individual `.md` files in `stock-analysis/` (plus `.xlsx`
   where modeling is requested).

---

*Generated with the Anthropic Claude for Financial Services skill set
(`sector-overview`, `competitive-analysis`). Draft for review — not investment advice.*
