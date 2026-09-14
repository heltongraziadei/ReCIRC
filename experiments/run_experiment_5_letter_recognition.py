#!/usr/bin/env python
"""Letter Recognition: CRC, AA-CRC (objetivo oficial) e ReCIRC.

Cinco correções em relação à versão anterior deste experimento, todas no bloco
AA-CRC. As três primeiras explicam por que aquela versão devolvia risco marginal
de ~5% com alpha = 10%.

1. O ridge não penaliza mais o intercepto. A primeira coordenada da condição de
   estacionariedade do objetivo dos autores (phi_{i0} == 1 para todo i) é

       risco_empirico(C) = alpha - 1/n - 2 * lambda * theta_0,

   ou seja, penalizar theta_0 desloca o nível marginal do risco para baixo por
   exatamente 2 * lambda * theta_0. A penalização passa a incidir apenas nos
   coeficientes de inclinação e é aplicada FORA de J/J_prime, que são chamados
   com regularisation=None; o objetivo dos autores permanece intacto. A
   identidade acima é verificada em --self-check.

   lambda NÃO é reduzido: o termo de dados é linear por partes em u e decresce
   sem limite ao longo de direções que separam as observações de rank 1 das
   demais, saturando só no truncamento INF_BORN_INT do integrando. A penalidade
   é o único termo que mantém theta finito, e baixá-la leva a soluções bang-bang
   coladas na fronteira. O conserto é isentar theta_0, não encolher lambda. Se
   ainda assim a solução encostar no truncamento, `fit_original_aacrc` escala
   lambda por 10 até obter um ótimo interior e registra o valor usado em
   `ridge_used` / `ridge_path_tried`.

2. O score deixa de ser a grade de ranks por default. Com scores (K-r+1)/K
   idênticos para todo x, J é linear por partes com vértices nos K pontos da
   grade, o mínimo cai exatamente sobre um vértice e a pertinência passa a ser
   decidida por arredondamento de ponto flutuante em `score >= u`; o risco
   marginal trava num degrau da escada top-m (o erro top-3 deste classificador,
   ~5%) e não existe theta que atinja alpha = 10%. O default passa a ser o score
   APS: s(x, y) = 1 - massa acumulada até y na ordem decrescente de
   probabilidade, com +infinito no rank 1 para garantir conjuntos não vazios.
   Ele é contínuo, depende de x, é estritamente decrescente no rank e mantém a
   convenção dos autores (inclui quando score >= u). O modo 'rank' continua
   disponível via --aacrc-score rank para reproduzir o comportamento antigo.

3. Calibração escalar pós-ajuste, agora ligada por default. theta é ajustado em
   D_risk e um deslocamento b é calibrado por CRC em C, exatamente como o ReCIRC
   ajusta o modelo de risco em D_risk e calibra o orçamento em C. Como
   perda_i(b) = 1{s_i(y_i) - u_i < b}, o b ótimo é um quantil empírico exato com
   a correção (n+1), sem grade: b = d_(j+1) com j = floor(alpha*(n+1)) - 1. Isso
   devolve ao AA-CRC a garantia marginal R <= alpha que a versão anterior havia
   removido, e ancora o nível marginal independentemente de vieses do otimizador.

4. Deslocamento randomizado opcional (--aacrc-randomized-offset). Em famílias
   discretas o risco alcançável é uma escada; sorteando b_lo com probabilidade
   gamma e b_hi = próximo ponto de quebra com probabilidade 1 - gamma, com gamma
   resolvendo (n/(n+1)) * (gamma*R_lo + (1-gamma)*R_hi) + 1/(n+1) = alpha, o
   nível marginal atinge alpha essencialmente na igualdade.

5. Desempate explícito. `top_m_from_scores` aceita uma tolerância; o default
   'inclusive' (1e-9) resolve empates a favor do conjunto maior, de forma
   determinística e conservadora, em vez de deixar o resultado à mercê do
   arredondamento. A verificação de equivalência com a codificação dos autores
   continua sendo feita com comparação estrita (convenção deles) e o diagnóstico
   passa a reportar a fração de limiares colados em um vértice da grade.

Ponto de comparabilidade: theta agora é ajustado em D_risk (30%) e calibrado em
C (20%), o mesmo orçamento de dados do ReCIRC; T (20%) é usado só para avaliação.
Com --aacrc-fit-split cal, C é dividido em ajuste/calibração.

Chama J/J_prime dos autores no commit fixado, via SLSQP. Splits estratificados e
grade de 81 pontos do ReCIRC herdados do script anterior. --self-check exige
apenas NumPy/SciPy. Nenhum pacote é instalado automaticamente.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize

AACRC_COMMIT = "64504c011ac2db910e258037e48170a63381b5e6"
AACRC_SOURCE_SHA256 = "9b3cffcb45f2e9a74f467ec94a00be7fcf2a24ec85f2dfe56f01d7fbd4a51315"
AACRC_REPO = Path(__file__).resolve().parents[1] / "AA-CRC"
AACRC_INTEGRATION = "serial"
AACRC_RIDGE = 1e-2            # correção 1: mesma escala de antes, porém sem penalizar o intercepto
AACRC_MAXITER = 200
AACRC_SCORE = "aps"           # correção 2: 'aps' (contínuo) ou 'rank' (grade discreta, legado)
AACRC_FIT_SPLIT = "risk"      # 'risk' => theta em D_risk e b em C; 'cal' => divide C
AACRC_CALIB_FRAC = 0.5        # usado apenas quando AACRC_FIT_SPLIT == 'cal'
AACRC_POSTFIT = True          # correção 3
AACRC_RANDOMIZED_OFFSET = False  # correção 4
AACRC_TIE = "inclusive"       # correção 5: 'inclusive' ou 'strict'
AACRC_TIE_TOL = 1e-9
AACRC_MAX_FIT_ROWS = None
AACRC_OUTPUT_DIR = None
AACRC_MODULE = None

SCORE_TIEBREAK = 1e-9         # torna os scores APS estritamente decrescentes no rank


def load_original_aacrc(repo, integration="serial"):
    """Load the pinned authors' module; optionally select its own serial helper."""
    if integration not in {"serial", "parallel"}:
        raise ValueError("integration must be 'serial' or 'parallel'.")
    repo = Path(repo).resolve()
    path = repo / "multiaccurate_cp" / "utils" / "multiaccurate.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"AA-CRC source missing: {path}. Clone vincentblot28/AA-CRC, "
            f"checkout {AACRC_COMMIT}, and pass --aacrc-repo PATH."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != AACRC_SOURCE_SHA256:
        raise RuntimeError(f"Unexpected AA-CRC source SHA-256: {digest}; expected {AACRC_SOURCE_SHA256}.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    module = importlib.import_module("multiaccurate_cp.utils.multiaccurate")
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("Another multiaccurate_cp module is already loaded; use a fresh Python process.")
    if not hasattr(module, "_original_parallel_integrals"):
        module._original_parallel_integrals = module._I_vec_multi_proc
    module._I_vec_multi_proc = (
        module._I_vec_multi_proc2 if integration == "serial" else module._original_parallel_integrals
    )
    return module


# -----------------------------------------------------------------------------
# Scores de conformidade (correção 2)
# -----------------------------------------------------------------------------


def rank_score_grid(K):
    """Grade discreta legado: scores decrescentes por rank, rank 1 sempre incluído."""
    if K < 2:
        raise ValueError("Need at least two classes.")
    scores = (K - np.arange(1, K + 1, dtype=float) + 1) / K
    scores[0] = np.inf
    return scores


def conformity_scores(proba, mode=AACRC_SCORE, tiebreak=SCORE_TIEBREAK):
    """Matriz (n, K) de scores em ordem decrescente de probabilidade.

    Convenção dos autores: score maior => mais conforme; inclui quando score >= u.
    A coluna 0 (rank 1) recebe +infinito, o que garante conjuntos não vazios e
    tamanho em 1..K, exatamente como na versão de rank.

    mode='aps'  : s_r = 1 - (massa acumulada até o rank r), contínuo e dependente
                  de x, com um desempate infinitesimal que o torna estritamente
                  decrescente em r mesmo quando há probabilidades nulas na cauda.
    mode='rank' : s_r = (K - r + 1)/K, a grade discreta compartilhada por todo x.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("Expected a two-dimensional probability matrix.")
    n, K = proba.shape
    if K < 2:
        raise ValueError("Need at least two classes.")
    if mode == "rank":
        S = np.tile(rank_score_grid(K), (n, 1))
        return S
    if mode != "aps":
        raise ValueError("aacrc score mode must be 'aps' or 'rank'.")
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("Expected finite nonnegative probabilities.")
    sortp = np.sort(proba, axis=1)[:, ::-1]
    cum = np.cumsum(sortp, axis=1)
    S = np.clip(1.0 - cum, 0.0, 1.0)
    S = S + (K - np.arange(1, K + 1, dtype=float))[None, :] * tiebreak
    S[:, 0] = np.inf
    if not (np.diff(S[:, 1:], axis=1) < 0).all():
        raise ValueError("Conformity scores must be strictly decreasing in rank.")
    return S


def top_m_from_scores(u, S, tie_tol=0.0):
    """Tamanho do conjunto: número de scores da linha i que atingem o limiar u_i.

    tie_tol > 0 resolve empates a favor do conjunto maior (correção 5); com
    tie_tol = 0 recupera-se a comparação estrita da codificação dos autores.
    """
    u = np.asarray(u, dtype=float)
    S = np.asarray(S, dtype=float)
    if u.ndim != 1 or not np.isfinite(u).all():
        raise ValueError("Expected finite one-dimensional score thresholds.")
    if S.ndim != 2 or len(S) != len(u):
        raise ValueError("Score matrix must have one row per threshold.")
    if tie_tol < 0:
        raise ValueError("tie_tol must be nonnegative.")
    return (S >= u[:, None] - tie_tol).sum(axis=1).astype(int)


def true_scores_from_ranks(S, ranks):
    """Score do rótulo verdadeiro de cada observação."""
    ranks = np.asarray(ranks, dtype=int)
    if ranks.ndim != 1 or len(ranks) != len(S):
        raise ValueError("Expected one rank per row of the score matrix.")
    if not np.isin(ranks, np.arange(1, S.shape[1] + 1)).all():
        raise ValueError("True-label ranks must be integers in 1..K.")
    return S[np.arange(len(S)), ranks - 1]


def encode_loss(S, ranks):
    """Rótulos one-hot em coordenadas de rank; perda de falso negativo = erro top-m."""
    S = np.asarray(S, dtype=float)
    K = S.shape[1]
    ranks = np.asarray(ranks, dtype=int)
    if ranks.ndim != 1 or len(ranks) != len(S) or not np.isin(ranks, np.arange(1, K + 1)).all():
        raise ValueError("True-label ranks must be integers in 1..K, one per row.")
    labels = np.eye(K)[ranks - 1]
    return [row[:, None] for row in labels], [row[:, None].copy() for row in S]


def vertex_pinning_rate(u, S, tol=1e-9):
    """Fração de limiares colados em um vértice de J (diagnóstico da correção 2)."""
    u = np.asarray(u, dtype=float)
    finite = np.where(np.isfinite(S), S, np.nan)
    gap = np.nanmin(np.abs(finite - u[:, None]), axis=1)
    return float(np.mean(gap <= tol))


# -----------------------------------------------------------------------------
# Objetivo dos autores com ridge apenas nas inclinações (correção 1)
# -----------------------------------------------------------------------------


def make_objective(module, labels, scores, phi, alpha, n, ridge):
    """J/J_prime dos autores com regularisation=None mais ridge fora do intercepto.

    A penalidade é somada externamente para que o objetivo original permaneça
    inalterado e para que theta_0 — a única coordenada que fixa o nível marginal
    do risco — fique livre.
    """
    mask = np.ones(phi.shape[1], dtype=float)
    mask[0] = 0.0

    def objective(theta):
        base = module.J(theta, labels, scores, phi, alpha, n, None, 0.0)
        return float(base + ridge * np.sum(mask * theta ** 2))

    def gradient(theta):
        base = module.J_prime(theta, labels, scores, phi, alpha, n, None, 0.0)
        return np.asarray(base, dtype=float) + 2.0 * ridge * mask * theta

    return objective, gradient


def fit_original_aacrc(F_risk, F_fit, S_fit, ranks_fit, alpha=0.1,
                       ridge=AACRC_RIDGE, maxiter=200, module=None,
                       diagnostic_path=None, tie_tol=0.0, ridge_escalations=3):
    """Ajusta theta por SLSQP; padroniza as features com as estatísticas de D_risk.

    Se o ótimo encostar no truncamento INF_BORN_INT do objetivo (solução
    bang-bang, sintoma de penalidade insuficiente para um termo de dados linear),
    lambda é multiplicado por 10 e o ajuste é repetido, até `ridge_escalations`
    vezes. O lambda efetivamente usado fica registrado no diagnóstico.
    """
    if module is None:
        module = load_original_aacrc(AACRC_REPO, AACRC_INTEGRATION)
    F_risk, F_fit = np.asarray(F_risk, float), np.asarray(F_fit, float)
    S_fit = np.asarray(S_fit, float)
    ranks_fit = np.asarray(ranks_fit, int)
    n, K = len(ranks_fit), S_fit.shape[1]
    if not 0 < alpha < 1 or n <= 1 / alpha:
        raise ValueError("Need 0 < alpha < 1 and a fitting split larger than 1/alpha.")
    if not np.isfinite(ridge) or ridge < 0 or maxiter < 1:
        raise ValueError("ridge must be finite and nonnegative; maxiter must be positive.")
    if ridge_escalations < 0:
        raise ValueError("ridge_escalations must be nonnegative.")
    if (F_risk.ndim != 2 or F_fit.ndim != 2 or len(F_risk) == 0
            or len(F_fit) != n or len(S_fit) != n or F_risk.shape[1] != F_fit.shape[1]
            or not np.isfinite(F_risk).all() or not np.isfinite(F_fit).all()):
        raise ValueError("Expected finite, compatible risk/fitting feature matrices.")
    center, scale = F_risk.mean(axis=0), F_risk.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    phi = np.column_stack([np.ones(n), (F_fit - center) / scale])
    labels, scores = encode_loss(S_fit, ranks_fit)
    s_true = true_scores_from_ranks(S_fit, ranks_fit)
    finite_true = s_true[np.isfinite(s_true)]
    theta_start = np.zeros(phi.shape[1])
    theta_start[0] = float(np.median(finite_true)) if finite_true.size else 0.5
    u_cap = 0.9 * module.INF_BORN_INT

    attempts = []
    start = time.time()
    accepted = None
    lam = float(ridge)
    for escalation in range(ridge_escalations + 1):
        theta = theta_start.copy()
        objective, gradient = make_objective(module, labels, scores, phi, alpha, n, lam)
        success, result = False, None
        for iterations in (maxiter, 3 * maxiter):
            result = minimize(objective, theta, method="SLSQP", jac=gradient,
                              tol=1e-8, options={"maxiter": iterations, "disp": False})
            success = bool(result.success and np.isfinite(result.x).all() and np.isfinite(result.fun))
            attempts.append({"ridge": lam, "success": success, "status": int(result.status),
                             "message": str(result.message), "iterations": int(result.nit),
                             "objective": float(result.fun) if np.isfinite(result.fun) else None})
            if success:
                break
            if np.isfinite(result.x).all():
                theta = result.x.copy()
        if not success:
            lam *= 10.0
            continue
        theta = result.x.copy()
        u = phi @ theta
        interior = bool(np.max(np.abs(u)) <= u_cap)
        attempts[-1].update({"max_abs_threshold": float(np.max(np.abs(u))), "interior": interior})
        if interior or escalation == ridge_escalations:
            accepted = (lam, theta, u, interior)
            break
        lam *= 10.0

    diag = {"source_commit": AACRC_COMMIT, "source_sha256": AACRC_SOURCE_SHA256,
            "ridge_on_intercept": False, "score_mode": AACRC_SCORE,
            "feature_standardization_split": "D_risk", "n_fit": n,
            "n_classes": K, "ridge_requested": float(ridge),
            "ridge_used": None, "ridge_path_tried": sorted({a["ridge"] for a in attempts}),
            "success": accepted is not None,
            "attempts": attempts, "elapsed_seconds": time.time() - start}
    success = accepted is not None
    if success:
        lam, theta, u, interior = accepted
        m = top_m_from_scores(u, S_fit, tie_tol=tie_tol)
        direct_loss = (ranks_fit > top_m_from_scores(u, S_fit, tie_tol=0.0)).astype(float)
        encoded_loss = module._I_prime_list(labels, scores, np.maximum(u, 0), alpha, n) + alpha - 1 / n
        error = float(np.max(np.abs(direct_loss - encoded_loss)))
        fit_risk = float((ranks_fit > m).mean())
        implied = float(alpha - 1 / n)
        diag.update({"ridge_used": float(lam), "theta": theta.tolist(),
                     "center": center.tolist(), "scale": scale.tolist(),
                     "fit_risk": fit_risk, "mean_set_size_fit": float(m.mean()),
                     "stationarity_implied_risk": implied,
                     "stationarity_gap": float(fit_risk - implied),
                     "encoding_max_abs_error": error,
                     "interior_solution": bool(interior),
                     "vertex_pinning_rate": vertex_pinning_rate(u, S_fit),
                     "threshold_min": float(u.min()), "threshold_max": float(u.max()),
                     "threshold_spread": float(u.max() - u.min())})
        if error > 1e-10 or np.any(u < -module.INF_BORN_INT):
            success = False
            diag.update(success=False, error="Loss encoding mismatch or original objective lower truncation reached.")
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps(diag, indent=2, allow_nan=False))
    if not success:
        raise RuntimeError("Original AA-CRC optimization failed: " + json.dumps(diag))
    if not diag["interior_solution"]:
        warnings.warn("AA-CRC: solução encostada no truncamento INF_BORN_INT mesmo após escalar o ridge; "
                      "aumente --aacrc-ridge.")
    return {"theta": theta, "center": center, "scale": scale, "diagnostics": diag}


def predict_original_aacrc(fit, F):
    phi = np.column_stack([np.ones(len(F)), (np.asarray(F, float) - fit["center"]) / fit["scale"]])
    return phi @ fit["theta"]


# -----------------------------------------------------------------------------
# Calibração escalar pós-ajuste por CRC (correções 3 e 4)
# -----------------------------------------------------------------------------


def calibrate_offset(u_cal, S_cal, ranks_cal, alpha, randomized=False):
    """Escolhe b por CRC exato: perda_i(b) = 1{d_i < b} com d_i = s_i(y_i) - u_i.

    O maior b admissível é o quantil empírico d_(j+1) com j = floor(alpha*(n+1)) - 1,
    que é exatamente a correção (n+1) do conformal split. Se randomized=True,
    devolve também o próximo ponto de quebra e o peso gamma que faz o limite CRC
    valer alpha na igualdade.
    """
    u_cal = np.asarray(u_cal, float)
    d = np.sort(true_scores_from_ranks(S_cal, ranks_cal) - u_cal)
    n = len(d)
    if not 0 < alpha < 1 or n <= 1 / alpha:
        raise ValueError("Calibration split must satisfy n > 1/alpha.")
    j = int(np.floor(alpha * (n + 1))) - 1
    if j < 0:
        raise ValueError("alpha*(n+1) < 1: no admissible offset.")
    j = min(j, n - 1)
    b_lo = float(d[j])
    risk_lo = float(np.mean(d < b_lo))
    bound_lo = (n / (n + 1.0)) * risk_lo + 1.0 / (n + 1.0)
    info = {"n_cal": n, "j": j, "b_lo": b_lo, "risk_lo": risk_lo, "crc_bound_lo": float(bound_lo),
            "randomized": bool(randomized), "b_hi": None, "risk_hi": None, "gamma": 1.0}
    if not randomized or j + 1 >= n:
        return info
    b_hi = float(d[j + 1])
    risk_hi = float(np.mean(d < b_hi))
    bound_hi = (n / (n + 1.0)) * risk_hi + 1.0 / (n + 1.0)
    if bound_hi <= alpha or not np.isfinite(b_hi) or risk_hi <= risk_lo:
        gamma = 0.0 if bound_hi <= alpha else 1.0
    else:
        # (n/(n+1)) * (gamma*risk_lo + (1-gamma)*risk_hi) + 1/(n+1) = alpha
        gamma = (bound_hi - alpha) / (bound_hi - bound_lo)
        gamma = float(np.clip(gamma, 0.0, 1.0))
    info.update({"b_hi": b_hi, "risk_hi": risk_hi, "crc_bound_hi": float(bound_hi), "gamma": gamma})
    return info


def apply_offset(u, offset, rng=None):
    """Aplica b_lo, ou sorteia entre b_lo e b_hi quando o deslocamento é randomizado."""
    u = np.asarray(u, float)
    if not offset["randomized"] or offset["b_hi"] is None:
        return u + offset["b_lo"]
    if rng is None:
        rng = np.random.default_rng(0)
    pick_lo = rng.random(len(u)) < offset["gamma"]
    b = np.where(pick_lo, offset["b_lo"], offset["b_hi"])
    return u + b


def check_original_aacrc(repo):
    """Verificações offline: identidade da perda, objetivo, gradiente, viés do
    intercepto, scores APS e validade do deslocamento calibrado."""
    module = load_original_aacrc(repo, "serial")
    K = 26

    # (a) grade de rank: identidade da perda em todos os ranks e nas fronteiras
    ranks = np.arange(1, K + 1)
    S_rank = np.tile(rank_score_grid(K), (K, 1))
    labels, scores = encode_loss(S_rank, ranks)
    grid = rank_score_grid(K)[1:]
    thresholds = np.concatenate([[-1, 0, 1, 2], grid,
                                 np.nextafter(grid, -np.inf), np.nextafter(grid, np.inf)])
    max_error = 0.0
    for threshold in thresholds:
        u = np.full(K, threshold)
        m = top_m_from_scores(u, S_rank)
        direct = (ranks > m).astype(float)
        encoded = module._I_prime_list(labels, scores, np.maximum(u, 0), 0.2, K) + 0.2 - 1 / K
        max_error = max(max_error, float(np.max(np.abs(direct - encoded))))
        assert np.all((m >= 1) & (m <= K))
        assert np.allclose(direct, encoded, atol=1e-12)
    assert top_m_from_scores(np.array([-1.0]), S_rank[:1])[0] == K
    assert top_m_from_scores(np.array([2.0]), S_rank[:1])[0] == 1
    assert np.all(np.diff(top_m_from_scores(np.sort(thresholds), np.tile(rank_score_grid(K), (len(thresholds), 1)))) <= 0)

    # (b) scores APS: estritamente decrescentes, tamanho em 1..K, mesma identidade
    rng = np.random.default_rng(11)
    proba = rng.dirichlet(np.full(K, 0.4), size=64)
    S_aps = conformity_scores(proba, mode="aps")
    assert np.isinf(S_aps[:, 0]).all() and np.isfinite(S_aps[:, 1:]).all()
    assert (np.diff(S_aps[:, 1:], axis=1) < 0).all()
    ranks_aps = rng.integers(1, K + 1, size=len(S_aps))
    labels_a, scores_a = encode_loss(S_aps, ranks_aps)
    aps_error = 0.0
    for level in (0.0, 0.05, 0.2, 0.5, 0.9):
        u = np.full(len(S_aps), level)
        m = top_m_from_scores(u, S_aps)
        assert np.all((m >= 1) & (m <= K))
        direct = (ranks_aps > m).astype(float)
        encoded = module._I_prime_list(labels_a, scores_a, np.maximum(u, 0), 0.2, len(S_aps)) + 0.2 - 1 / len(S_aps)
        aps_error = max(aps_error, float(np.max(np.abs(direct - encoded))))
        assert np.allclose(direct, encoded, atol=1e-12)

    # (c) objetivo e gradiente com ridge apenas nas inclinações
    n, alpha, ridge = 100, 0.2, 0.01
    F_risk, F_fit = rng.normal(size=(120, 4)), rng.normal(size=(n, 4))
    ranks = rng.integers(1, K + 1, size=n)
    S_fit = np.tile(rank_score_grid(K), (n, 1))
    labels, scores = encode_loss(S_fit, ranks)
    phi = np.column_stack([np.ones(n), F_fit])
    theta = np.array([0.4, 0.02, -0.01, 0.03, 0.01])
    u, target = phi @ theta, alpha - 1 / n
    true_scores = true_scores_from_ranks(S_fit, ranks)
    exact = np.mean(np.maximum(u - true_scores, 0) - u * target)
    exact -= module.INF_BORN_INT * target
    exact += ridge * np.sum(theta[1:] ** 2)
    objective, gradient = make_objective(module, labels, scores, phi, alpha, n, ridge)
    error = abs(float(objective(theta) - exact))
    assert error < 0.005, error
    loss = (ranks > top_m_from_scores(u, S_fit)).astype(float)
    expected = np.mean(phi * (loss - target)[:, None], axis=0) + 2 * ridge * np.r_[0.0, theta[1:]]
    assert np.allclose(gradient(theta), expected, atol=1e-12)
    assert abs(gradient(theta)[0] - (loss.mean() - target)) < 1e-12  # intercepto livre

    # (d) viés do intercepto: com ridge em theta_0 o risco ajustado cai 2*lambda*theta_0
    n_b, K_b, alpha_b = 2000, 8, 0.2
    rng_b = np.random.default_rng(3)
    proba_b = rng_b.dirichlet(np.full(K_b, 0.5), size=n_b)
    S_b = conformity_scores(proba_b, mode="aps")
    ranks_b = rng_b.integers(1, K_b + 1, size=n_b)
    phi_b = np.column_stack([np.ones(n_b), rng_b.normal(size=(n_b, 2))])
    labels_b, scores_b = encode_loss(S_b, ranks_b)
    s_true_b = true_scores_from_ranks(S_b, ranks_b)
    tau_b = alpha_b - 1 / n_b
    biases = {}
    for lam in (0.0, 0.05):
        f = lambda th: (np.mean(np.maximum(phi_b @ th - s_true_b, 0) - (phi_b @ th) * tau_b)
                        + lam * float(np.sum(th ** 2)))
        g = lambda th: (np.mean(phi_b * (((phi_b @ th) > s_true_b).astype(float) - tau_b)[:, None], axis=0)
                        + 2 * lam * th)
        r = minimize(f, np.r_[0.3, np.zeros(2)], jac=g, method="SLSQP", tol=1e-10, options={"maxiter": 400})
        risk = float(np.mean(ranks_b > top_m_from_scores(phi_b @ r.x, S_b)))
        biases[lam] = {"risk": risk, "predicted": float(tau_b - 2 * lam * r.x[0]), "theta0": float(r.x[0])}
        assert abs(risk - (tau_b - 2 * lam * r.x[0])) < 5e-3, biases  # viés = 2*lambda*theta_0

    # (e) deslocamento calibrado: limite CRC respeitado; randomizado atinge alpha
    u_cal = rng.normal(0.4, 0.2, size=n_b)
    off = calibrate_offset(u_cal, S_b, ranks_b, alpha_b, randomized=False)
    m_cal = top_m_from_scores(apply_offset(u_cal, off), S_b)
    assert off["crc_bound_lo"] <= alpha_b + 1e-12
    assert float(np.mean(ranks_b > m_cal)) <= alpha_b + 1e-12
    off_r = calibrate_offset(u_cal, S_b, ranks_b, alpha_b, randomized=True)
    assert 0.0 <= off_r["gamma"] <= 1.0

    fit = fit_original_aacrc(F_risk, F_fit, S_fit, ranks, alpha=alpha,
                             ridge=1e-2, module=module)
    assert predict_original_aacrc(fit, F_fit[:8]).shape == (8,)
    print(json.dumps({"loss_identity_max_error": max_error,
                      "aps_loss_identity_max_error": aps_error,
                      "boundary_checks": "passed",
                      "official_quadrature_abs_error": error,
                      "gradient_check": "passed",
                      "intercept_bias_check": biases,
                      "offset_calibration": {"crc_bound": off["crc_bound_lo"], "gamma": off_r["gamma"]},
                      "optimizer": fit["diagnostics"]}, indent=2))


if __name__ == "__main__" and "--self-check" in sys.argv:
    parser = argparse.ArgumentParser(description="Offline validation of the corrected AA-CRC block.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--aacrc-repo", type=Path, default=AACRC_REPO)
    args = parser.parse_args()
    check_original_aacrc(args.aacrc_repo)
    raise SystemExit(0)


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

ALPHA = 0.10
N_TRIALS = 10
BASE_SEED = 42

BASE_TRAIN_FRAC = 0.30
RISK_FRAC = 0.30
CAL_FRAC = 0.20
TEST_FRAC = 0.20

RISK_MODEL = "tabicl"  # "tabicl" or "hgb"
FALLBACK_TO_HGB_IF_TABICL_FAILS = False
K_AUG_PER_EXAMPLE = 8
MAX_RISK_TRAIN_ROWS = 6000
A_GRID = np.linspace(0.0, 1.0, 81)
A_GRID.setflags(write=False)
N_BINS = 5

# Detect Google Drive mounted (Colab) and use as default
if os.path.isdir("/content/drive"):
    OUT_DIR = "/content/drive/MyDrive/PythonReCIRC/results/experiment_5_letter_recognition_official_aacrc"
else:
    OUT_DIR = "letter_recognition_official_aacrc_recirc_results"

BASE_CLF_MAX_ITER = 1500
BASE_CLF_C = 2.0


# -----------------------------------------------------------------------------
# Mathematical helpers
# -----------------------------------------------------------------------------


def true_label_ranks(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Rank 1 => true class most likely; Rank K => least likely."""
    order = np.argsort(-proba, axis=1, kind="stable")
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(1, order.shape[1] + 1)
    return inv[np.arange(len(y_true)), y_true].astype(int)


def losses_from_m_values(ranks: np.ndarray, m_values: Iterable[int]) -> np.ndarray:
    m_values = np.asarray(m_values, dtype=int)
    return (ranks[:, None] > m_values[None, :]).astype(np.float32)


def losses_from_m_matrix(ranks: np.ndarray, m_matrix: np.ndarray) -> np.ndarray:
    return (ranks[:, None] > m_matrix).astype(np.float32)


def score_features(X_raw: np.ndarray, proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    sortp = np.sort(proba, axis=1)[:, ::-1]
    top5 = sortp[:, :5]
    entropy = -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1) / np.log(proba.shape[1])
    margin12 = sortp[:, 0] - sortp[:, 1]
    mass_top3 = sortp[:, :3].sum(axis=1)
    mass_top5 = sortp[:, :5].sum(axis=1)
    n_above_005 = (proba >= 0.05).sum(axis=1) / proba.shape[1]
    n_above_010 = (proba >= 0.10).sum(axis=1) / proba.shape[1]
    prob_std = proba.std(axis=1)

    return np.column_stack([
        X_raw.astype(np.float32),
        top5,
        entropy,
        margin12,
        mass_top3,
        mass_top5,
        n_above_005,
        n_above_010,
        prob_std,
    ]).astype(np.float32)


def difficulty_score_from_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    return -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1) / np.log(proba.shape[1])


def make_equal_mass_bins(score: np.ndarray, n_bins: int = 5) -> np.ndarray:
    score = np.asarray(score)
    ranks = pd.Series(score).rank(method="first").values
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(bins, dtype=int)


# -----------------------------------------------------------------------------
# CRC calibration
# -----------------------------------------------------------------------------


def crc_upper_bound(mean_loss: np.ndarray, n: int, B: float = 1.0) -> np.ndarray:
    return (n / (n + 1.0)) * mean_loss + B / (n + 1.0)


def select_smallest_protective_param(params: np.ndarray, losses_cal: np.ndarray, alpha: float = ALPHA, B: float = 1.0):
    params = np.asarray(params)
    n = losses_cal.shape[0]
    mean_losses = losses_cal.mean(axis=0)
    bounds = crc_upper_bound(mean_losses, n=n, B=B)
    ok = bounds <= alpha
    if ok.any():
        j = int(np.where(ok)[0][0])
    else:
        j = len(params) - 1
    path = pd.DataFrame({
        "param": params,
        "mean_cal_loss": mean_losses,
        "crc_bound": bounds,
        "passes": ok,
    })
    return params[j], j, path


def select_largest_budget_param(params: np.ndarray, losses_cal: np.ndarray, alpha: float = ALPHA, B: float = 1.0):
    params = np.asarray(params)
    n = losses_cal.shape[0]
    mean_losses = losses_cal.mean(axis=0)
    bounds = crc_upper_bound(mean_losses, n=n, B=B)
    ok = bounds <= alpha
    if ok.any():
        j = int(np.where(ok)[0][-1])
    else:
        j = 0
    path = pd.DataFrame({
        "param": params,
        "mean_cal_loss": mean_losses,
        "crc_bound": bounds,
        "passes": ok,
    })
    return params[j], j, path


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

LETTER_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
FEATURE_NAMES = [
    "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]


def load_letter_recognition(cache_file: str = "./data_letter_recognition/letter-recognition.csv"):
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(cache_file):
        print("Downloading Letter Recognition from UCI...")
        df = pd.read_csv(LETTER_URL, header=None)
        df.to_csv(cache_file, index=False, header=False)
    else:
        df = pd.read_csv(cache_file, header=None, dtype=str)
        if len(df):
            first = df.iloc[0].astype(str).tolist()
            if first in ([str(i) for i in range(17)], ["letter"] + FEATURE_NAMES):
                df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] != 17:
        raise ValueError(f"Esperava 17 colunas; vieram {df.shape[1]}")

    df.columns = ["letter"] + FEATURE_NAMES
    if len(df) == 0 or not df["letter"].astype(str).str.fullmatch("[A-Z]").all():
        raise ValueError("Esperava observações com letras A-Z na primeira coluna.")
    le = LabelEncoder()
    y = le.fit_transform(df["letter"].astype(str).values)
    X = df[FEATURE_NAMES].astype(np.float32).values
    if not np.isfinite(X).all():
        raise ValueError("As covariáveis devem ser finitas.")
    class_names = list(le.classes_)
    return X, y, class_names, df


# -----------------------------------------------------------------------------
# Splits and base classifier
# -----------------------------------------------------------------------------


def make_four_way_split(y: np.ndarray, seed: int, base_train_frac: float = BASE_TRAIN_FRAC, risk_frac: float = RISK_FRAC,
                       cal_frac: float = CAL_FRAC, test_frac: float = TEST_FRAC):
    idx = np.arange(len(y))

    idx_base, idx_rest = train_test_split(
        idx,
        train_size=base_train_frac,
        random_state=seed,
        stratify=y,
    )

    rest_frac = 1.0 - base_train_frac
    risk_rel = risk_frac / rest_frac
    idx_risk, idx_rest2 = train_test_split(
        idx_rest,
        train_size=risk_rel,
        random_state=seed + 17,
        stratify=y[idx_rest],
    )

    cal_rel = cal_frac / (cal_frac + test_frac)
    idx_cal, idx_test = train_test_split(
        idx_rest2,
        train_size=cal_rel,
        random_state=seed + 31,
        stratify=y[idx_rest2],
    )

    return idx_base, idx_risk, idx_cal, idx_test


def fit_base_classifier(X: np.ndarray, y: np.ndarray, idx_base: np.ndarray, seed: int, K: int):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=BASE_CLF_C,
            max_iter=BASE_CLF_MAX_ITER,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    clf.fit(X[idx_base], y[idx_base])

    raw = clf.predict_proba(X)
    classes = clf.named_steps["logisticregression"].classes_.astype(int)
    proba = np.zeros((len(X), K), dtype=np.float32)
    proba[:, classes] = raw.astype(np.float32)
    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    return clf, proba


# -----------------------------------------------------------------------------
# AA-CRC with the authors' objective
# -----------------------------------------------------------------------------


def run_original_aacrc(F_risk, ranks_risk, S_risk, F_cal, ranks_cal, S_cal,
                       F_test, ranks_test, S_test, trial, seed, idx_test, alpha=ALPHA):
    """AA-CRC: theta pelo objetivo dos autores, b por CRC, avaliação em T.

    Com --aacrc-fit-split risk (default) theta é ajustado em D_risk e b calibrado
    em C — o mesmo orçamento de dados do ReCIRC. Com 'cal', C é dividido entre
    ajuste e calibração, de modo que o conjunto usado no deslocamento continua
    sendo disjunto do usado no ajuste.
    """
    rng = np.random.default_rng(seed + 991)

    if AACRC_FIT_SPLIT == "risk":
        F_fit, ranks_fit, S_fit = F_risk, ranks_risk, S_risk
        F_off, ranks_off, S_off = F_cal, ranks_cal, S_cal
    elif AACRC_FIT_SPLIT == "cal":
        perm = np.random.default_rng(seed + 7).permutation(len(ranks_cal))
        n_fit = int(round((1.0 - AACRC_CALIB_FRAC) * len(perm)))
        i_fit, i_off = perm[:n_fit], perm[n_fit:]
        if min(len(i_fit), len(i_off)) <= 1 / alpha:
            raise ValueError("Ambas as partes de C precisam ter mais que 1/alpha observações.")
        F_fit, ranks_fit, S_fit = F_cal[i_fit], ranks_cal[i_fit], S_cal[i_fit]
        F_off, ranks_off, S_off = F_cal[i_off], ranks_cal[i_off], S_cal[i_off]
    else:
        raise ValueError("aacrc fit split must be 'risk' or 'cal'.")

    if AACRC_MAX_FIT_ROWS is not None and len(ranks_fit) > AACRC_MAX_FIT_ROWS:
        take = np.random.default_rng(seed + 13).choice(len(ranks_fit), size=AACRC_MAX_FIT_ROWS, replace=False)
        F_fit, ranks_fit, S_fit = F_fit[take], ranks_fit[take], S_fit[take]

    tie_tol = AACRC_TIE_TOL if AACRC_TIE == "inclusive" else 0.0
    diagnostic_path = (AACRC_OUTPUT_DIR / f"aacrc_diagnostics_trial_{trial:03d}.json"
                       if AACRC_OUTPUT_DIR is not None else None)
    fit = fit_original_aacrc(F_risk, F_fit, S_fit, ranks_fit, alpha=alpha,
                             ridge=AACRC_RIDGE, maxiter=AACRC_MAXITER,
                             module=AACRC_MODULE, diagnostic_path=None, tie_tol=tie_tol)

    u_test_raw = predict_original_aacrc(fit, F_test)
    if AACRC_POSTFIT:
        u_off = predict_original_aacrc(fit, F_off)
        offset = calibrate_offset(u_off, S_off, ranks_off, alpha,
                                  randomized=AACRC_RANDOMIZED_OFFSET)
        u_test = apply_offset(u_test_raw, offset, rng=rng)
        m_off = top_m_from_scores(apply_offset(u_off, offset, rng=np.random.default_rng(seed + 992)),
                                  S_off, tie_tol=tie_tol)
        offset_diag = dict(offset)
        offset_diag["risk_on_offset_split"] = float((ranks_off > m_off).mean())
    else:
        offset = {"randomized": False, "b_lo": 0.0, "b_hi": None, "gamma": 1.0}
        u_test = u_test_raw
        offset_diag = {"applied": False}

    m = top_m_from_scores(u_test, S_test, tie_tol=tie_tol)
    losses = (ranks_test > m).astype(float)

    diag = dict(fit["diagnostics"])
    diag.update({
        "fit_split": AACRC_FIT_SPLIT,
        "n_offset_split": int(len(ranks_off)),
        "postfit_scalar_calibration": bool(AACRC_POSTFIT),
        "randomized_offset": bool(AACRC_RANDOMIZED_OFFSET),
        "tie_handling": AACRC_TIE,
        "offset": offset_diag,
        "test_vertex_pinning_rate": vertex_pinning_rate(u_test, S_test),
        "test_threshold_spread": float(u_test.max() - u_test.min()),
        "test_risk": float(losses.mean()),
        "test_mean_set_size": float(m.mean()),
    })
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps(diag, indent=2, allow_nan=False))
        pd.DataFrame({"test_index": idx_test, "threshold_u_raw": u_test_raw,
                      "threshold_u": u_test, "set_size": m,
                      "true_label_rank": ranks_test, "loss": losses}).to_csv(
            AACRC_OUTPUT_DIR / f"aacrc_predictions_trial_{trial:03d}.csv", index=False)
    return {
        "method": "aa_crc_official", "selected_param": float(offset.get("b_lo", 0.0)),
        "selected_m_mean": float(m.mean()), "test_risk": float(losses.mean()),
        "avg_set_size": float(m.mean()), "median_set_size": float(np.median(m)),
        "losses_test": losses, "sizes_test": m.astype(float),
    }


# -----------------------------------------------------------------------------
# ReCIRC
# -----------------------------------------------------------------------------


def make_tabicl_regressor():
    try:
        from tabicl import TabICLRegressor
    except Exception:
        try:
            from tabicl.sklearn import TabICLRegressor
        except Exception as e:
            raise ImportError("Não consegui importar TabICLRegressor. Tente: pip install tabicl") from e

    try:
        return TabICLRegressor()
    except TypeError:
        return TabICLRegressor


def build_augmented_risk_data(F: np.ndarray, ranks: np.ndarray, rng: np.random.Generator,
                             k_aug: int = K_AUG_PER_EXAMPLE, max_rows: int = MAX_RISK_TRAIN_ROWS, K_grid: np.ndarray = None):
    if K_grid is None:
        K_grid = np.arange(1, F.shape[0] if False else 0, dtype=int)
    # For this experiment, the m space is 1..K.
    # Keep the grid visible at runtime using the global K.
    global K
    m_grid = np.arange(1, K + 1, dtype=int)
    n = len(ranks)
    sampled_m = rng.choice(m_grid, size=(n, k_aug), replace=True)

    F_rep = np.repeat(F, k_aug, axis=0)
    m_flat = sampled_m.ravel().astype(int)
    rho_flat = (m_flat / K).astype(np.float32)
    y_aug = (np.repeat(ranks, k_aug) > m_flat).astype(np.float32)
    X_aug = np.column_stack([F_rep, rho_flat]).astype(np.float32)

    if max_rows is not None and len(X_aug) > max_rows:
        take = rng.choice(len(X_aug), size=max_rows, replace=False)
        X_aug = X_aug[take]
        y_aug = y_aug[take]

    return X_aug, y_aug


def fit_risk_estimator(F_risk: np.ndarray, ranks_risk: np.ndarray, seed: int, K_value: int = None, risk_model: str = RISK_MODEL):
    if K_value is None:
        K_value = F_risk.shape[0] if False else 0
    global K
    if K_value:
        K_local = K_value
    else:
        K_local = K

    rng = np.random.default_rng(seed)
    X_aug, y_aug = build_augmented_risk_data(F_risk, ranks_risk, rng, k_aug=K_AUG_PER_EXAMPLE, max_rows=MAX_RISK_TRAIN_ROWS, K_grid=np.arange(1, K_local + 1))

    scaler = StandardScaler()
    X_aug_s = scaler.fit_transform(X_aug)

    if risk_model == "tabicl":
        try:
            model = make_tabicl_regressor()
            model.fit(X_aug_s, y_aug)
            model_name = "tabicl"
        except Exception as e:
            if not FALLBACK_TO_HGB_IF_TABICL_FAILS:
                raise
            warnings.warn(f"TabICL falhou; usando HGB. Erro: {repr(e)}")
            model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            )
            model.fit(X_aug_s, y_aug)
            model_name = "hgb_fallback"
    elif risk_model == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        )
        model.fit(X_aug_s, y_aug)
        model_name = "hgb"
    else:
        raise ValueError("risk_model must be 'tabicl' or 'hgb'.")

    return {"model": model, "scaler": scaler, "model_name": model_name}


def predict_risk_table(estimator: Dict, F_query: np.ndarray, batch_size: int = 5000):
    n = len(F_query)
    global K
    m_grid = np.arange(1, K + 1, dtype=int)
    F_rep = np.repeat(F_query, K, axis=0)
    rho = np.tile(m_grid / K, n).astype(np.float32)
    Xq = np.column_stack([F_rep, rho]).astype(np.float32)
    Xq_s = estimator["scaler"].transform(Xq)

    preds = []
    model = estimator["model"]
    for start in range(0, len(Xq_s), batch_size):
        end = start + batch_size
        preds.append(np.asarray(model.predict(Xq_s[start:end]), dtype=np.float32))
    risk = np.concatenate(preds).reshape(n, K)
    risk = np.clip(risk, 0.0, 1.0)
    risk[:, -1] = 0.0
    risk = np.minimum.accumulate(risk, axis=1)
    return risk


def m_from_risk_budget(risk_table: np.ndarray, a_grid: np.ndarray):
    n, k = risk_table.shape
    out = np.empty((n, len(a_grid)), dtype=int)
    for j, a in enumerate(a_grid):
        ok = risk_table <= a
        first = np.argmax(ok, axis=1)
        no_ok = ~ok.any(axis=1)
        first[no_ok] = k - 1
        out[:, j] = np.arange(1, k + 1)[first]
    return out


try:
    from .risk_calibration import RiskCalibration, bin_groups
except ImportError:
    from risk_calibration import RiskCalibration, bin_groups


def run_recirc(F_risk: np.ndarray, ranks_risk: np.ndarray, F_cal: np.ndarray, ranks_cal: np.ndarray,
               F_test: np.ndarray, ranks_test: np.ndarray, trial: int, seed: int, alpha: float = ALPHA,
               risk_model: str = RISK_MODEL):
    estimator = fit_risk_estimator(F_risk, ranks_risk, seed=seed, risk_model=risk_model)

    risk_cal = predict_risk_table(estimator, F_cal)
    m_cal_by_a = m_from_risk_budget(risk_cal, A_GRID)
    losses_cal = losses_from_m_matrix(ranks_cal, m_cal_by_a)

    a_hat, j_hat, path = select_largest_budget_param(A_GRID, losses_cal, alpha=alpha)
    path["trial"] = trial
    path["method"] = f"recirc_{estimator['model_name']}"

    risk_test = predict_risk_table(estimator, F_test)
    m_test = m_from_risk_budget(risk_test, np.asarray([a_hat])).ravel()
    losses_test = (ranks_test > m_test).astype(np.float32)

    return {
        "method": f"recirc_{estimator['model_name']}",
        "selected_param": float(a_hat),
        "selected_m_mean": float(np.mean(m_test)),
        "test_risk": float(np.mean(losses_test)),
        "avg_set_size": float(np.mean(m_test)),
        "median_set_size": float(np.median(m_test)),
        "losses_test": losses_test,
        "sizes_test": m_test.astype(float),
        "calibration_path": path,
        "test_risk_table": risk_test,
    }


# -----------------------------------------------------------------------------
# Standard CRC and evaluation by bins
# -----------------------------------------------------------------------------


def run_standard_crc(ranks_cal: np.ndarray, ranks_test: np.ndarray, trial: int, K: int):
    losses_cal = losses_from_m_values(ranks_cal, np.arange(1, K + 1, dtype=int))
    m_hat, j_hat, path = select_smallest_protective_param(np.arange(1, K + 1, dtype=int), losses_cal, alpha=ALPHA)
    path["trial"] = trial
    path["method"] = "crc"

    m_test = np.full(len(ranks_test), int(m_hat), dtype=int)
    losses_test = (ranks_test > m_test).astype(np.float32)

    return {
        "method": "crc",
        "selected_param": int(m_hat),
        "selected_m_mean": float(m_hat),
        "test_risk": float(np.mean(losses_test)),
        "avg_set_size": float(m_hat),
        "median_set_size": float(m_hat),
        "losses_test": losses_test,
        "sizes_test": m_test.astype(float),
        "calibration_path": path,
    }


def summarize_bins(method_result: Dict, bins: np.ndarray, trial: int):
    rows = []
    losses = method_result["losses_test"]
    sizes = method_result["sizes_test"]
    for b in np.unique(bins):
        mask = bins == b
        rows.append({
            "trial": trial,
            "method": method_result["method"],
            "bin": int(b),
            "n": int(mask.sum()),
            "bin_risk": float(np.mean(losses[mask])),
            "avg_set_size": float(np.mean(sizes[mask])),
            "median_set_size": float(np.median(sizes[mask])),
        })
    return rows


def add_summary_row(method_result: Dict, trial: int, base_acc: float, test_acc: float):
    return {
        "trial": trial,
        "method": method_result["method"],
        "alpha": ALPHA,
        "selected_param": method_result["selected_param"],
        "selected_m_mean": method_result["selected_m_mean"],
        "test_risk": method_result["test_risk"],
        "excess_risk_event": float(method_result["test_risk"] > ALPHA),
        "avg_set_size": method_result["avg_set_size"],
        "median_set_size": method_result["median_set_size"],
        "base_train_accuracy": base_acc,
        "base_test_accuracy": test_acc,
    }


# -----------------------------------------------------------------------------
# Single trial loop
# -----------------------------------------------------------------------------


def run_one_trial(trial: int, seed: int, X: np.ndarray, y: np.ndarray, K: int, risk_model: str = RISK_MODEL, use_recirc: bool = True, diagnostic=None):
    idx_base, idx_risk, idx_cal, idx_test = make_four_way_split(y, seed)

    clf, proba = fit_base_classifier(X, y, idx_base, seed=seed, K=K)
    y_pred = proba.argmax(axis=1)
    base_acc = accuracy_score(y[idx_base], y_pred[idx_base])
    test_acc = accuracy_score(y[idx_test], y_pred[idx_test])

    ranks = true_label_ranks(proba, y)
    F = score_features(X, proba)
    S = conformity_scores(proba, mode=AACRC_SCORE)

    F_risk, ranks_risk, S_risk = F[idx_risk], ranks[idx_risk], S[idx_risk]
    F_cal, ranks_cal, S_cal = F[idx_cal], ranks[idx_cal], S[idx_cal]
    F_test, ranks_test, S_test = F[idx_test], ranks[idx_test], S[idx_test]

    difficulty = difficulty_score_from_proba(proba[idx_test])
    bins = make_equal_mass_bins(difficulty, n_bins=N_BINS)

    results = []
    paths = []

    res_crc = run_standard_crc(ranks_cal, ranks_test, trial, K)
    results.append(res_crc)
    paths.append(res_crc["calibration_path"])

    res_aa = run_original_aacrc(F_risk, ranks_risk, S_risk, F_cal, ranks_cal, S_cal,
                                F_test, ranks_test, S_test, trial, seed, idx_test, alpha=ALPHA)
    results.append(res_aa)

    if use_recirc:
        res_recirc = run_recirc(F_risk, ranks_risk, F_cal, ranks_cal, F_test, ranks_test, trial, seed=seed + 1000, risk_model=risk_model, alpha=ALPHA)
        if diagnostic is not None:
            losses_by_budget = losses_from_m_matrix(
                ranks_test, m_from_risk_budget(res_recirc["test_risk_table"], A_GRID))
            diagnostic.add_trial(A_GRID, losses_by_budget, bin_groups(bins, N_BINS),
                                 trial=trial, seed=seed, a_hat=res_recirc["selected_param"])
        results.append(res_recirc)
        paths.append(res_recirc["calibration_path"])

    summary_rows = []
    bin_rows = []
    for res in results:
        summary_rows.append(add_summary_row(res, trial, base_acc, test_acc))
        bin_rows.extend(summarize_bins(res, bins, trial))

    return summary_rows, bin_rows, paths


# -----------------------------------------------------------------------------
# Aggregation and plots
# -----------------------------------------------------------------------------


def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if len(summary_df) == 0:
        return pd.DataFrame()
    return (
        summary_df.groupby("method", as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            risk_mean=("test_risk", "mean"),
            risk_sd=("test_risk", "std"),
            excess_event_rate=("excess_risk_event", "mean"),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_sd=("avg_set_size", "std"),
            selected_m_mean=("selected_m_mean", "mean"),
            base_test_acc_mean=("base_test_accuracy", "mean"),
        )
        .sort_values("risk_mean")
    )


def aggregate_bins(bin_df: pd.DataFrame) -> pd.DataFrame:
    if len(bin_df) == 0:
        return pd.DataFrame()
    return (
        bin_df.groupby(["method", "bin"], as_index=False)
        .agg(
            bin_risk_mean=("bin_risk", "mean"),
            bin_risk_sd=("bin_risk", "std"),
            avg_set_size_mean=("avg_set_size", "mean"),
            n_mean=("n", "mean"),
        )
    )


def save_plots(summary_rows: List[Dict], bin_rows: List[Dict], output_dir: str, alpha: float = ALPHA):
    summary_df = pd.DataFrame(summary_rows)
    bin_df = pd.DataFrame(bin_rows)
    agg = aggregate_summary(summary_df)
    bins_agg = aggregate_bins(bin_df)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    for method, sub in summary_df.groupby("method"):
        sub = sub.sort_values("trial")
        plt.plot(sub["trial"], sub["test_risk"], marker="o", label=method)
    plt.axhline(alpha, linestyle="--", linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Marginal test risk")
    plt.title("Risco marginal por trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_risk_by_trial.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    methods = agg["method"].tolist()
    xpos = np.arange(len(methods))
    plt.bar(xpos, agg["risk_mean"].values)
    if len(agg) > 0:
        yerr = agg["risk_sd"].fillna(0).values / np.sqrt(np.maximum(agg["n_trials"].values, 1))
        plt.errorbar(xpos, agg["risk_mean"].values, yerr=yerr, fmt="none", capsize=4)
    plt.axhline(alpha, linestyle="--", linewidth=1)
    plt.xticks(xpos, methods, rotation=20, ha="right")
    plt.ylabel("Mean test risk")
    plt.title("Risco marginal médio acumulado")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_risk_by_method.png"), dpi=160)
    plt.close()

    if len(bins_agg):
        plt.figure(figsize=(8, 4.5))
        for method, sub in bins_agg.groupby("method"):
            sub = sub.sort_values("bin")
            plt.plot(sub["bin"], sub["bin_risk_mean"], marker="o", label=method)
        plt.axhline(alpha, linestyle="--", linewidth=1)
        plt.xlabel("Difficulty bin")
        plt.ylabel("Mean bin risk")
        plt.title("Risco por bin de dificuldade")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bin_risk_curves.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        for method, sub in bins_agg.groupby("method"):
            sub = sub.sort_values("bin")
            plt.plot(sub["bin"], sub["avg_set_size_mean"], marker="o", label=method)
        plt.xlabel("Difficulty bin")
        plt.ylabel("Average set size")
        plt.title("Tamanho médio por bin de dificuldade")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bin_set_size_curves.png"), dpi=160)
        plt.close()


def build_final_summary(summary_df: pd.DataFrame, bin_df: pd.DataFrame):
    summary_by_method = (
        summary_df.groupby("method", as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            risk_mean=("test_risk", "mean"),
            risk_sd=("test_risk", "std"),
            risk_se=("test_risk", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            excess_event_rate=("excess_risk_event", "mean"),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_sd=("avg_set_size", "std"),
            selected_m_mean=("selected_m_mean", "mean"),
            selected_m_sd=("selected_m_mean", "std"),
            base_test_acc_mean=("base_test_accuracy", "mean"),
        )
    )

    worst_bins = (
        bin_df.groupby(["trial", "method"], as_index=False)
        .agg(
            worst_bin_risk=("bin_risk", "max"),
            q90_bin_risk=("bin_risk", lambda x: np.quantile(x, 0.90)),
            q95_bin_risk=("bin_risk", lambda x: np.quantile(x, 0.95)),
            mean_positive_excess_bin=("bin_risk", lambda x: np.mean(np.maximum(np.asarray(x) - ALPHA, 0))),
        )
    )

    worst_by_method = (
        worst_bins.groupby("method", as_index=False)
        .agg(
            worst_bin_mean=("worst_bin_risk", "mean"),
            worst_bin_sd=("worst_bin_risk", "std"),
            q90_bin_mean=("q90_bin_risk", "mean"),
            q95_bin_mean=("q95_bin_risk", "mean"),
            mean_positive_excess_bin=("mean_positive_excess_bin", "mean"),
        )
    )

    summary_by_method = summary_by_method.merge(worst_by_method, on="method", how="left")
    summary_by_method = summary_by_method.sort_values("risk_mean")

    bins_by_method = (
        bin_df.groupby(["method", "bin"], as_index=False)
        .agg(
            bin_risk_mean=("bin_risk", "mean"),
            bin_risk_sd=("bin_risk", "std"),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_sd=("avg_set_size", "std"),
            n_mean=("n", "mean"),
        )
    )

    compact_table = summary_by_method.copy()
    compact_table["marginal_risk"] = compact_table.apply(
        lambda r: f"{r['risk_mean']:.4f} ± {r['risk_sd']:.4f}", axis=1
    )
    compact_table["worst_bin_risk"] = compact_table.apply(
        lambda r: f"{r['worst_bin_mean']:.4f} ± {r['worst_bin_sd']:.4f}", axis=1
    )
    compact_table["avg_set_size"] = compact_table.apply(
        lambda r: f"{r['avg_set_size_mean']:.2f} ± {r['avg_set_size_sd']:.2f}", axis=1
    )
    compact_table = compact_table[[
        "method", "n_trials", "marginal_risk", "excess_event_rate", "worst_bin_risk", "avg_set_size"
    ]]

    return summary_by_method, bins_by_method, compact_table


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento 5: Letter Recognition — CRC vs AA-CRC (objetivo oficial) vs ReCIRC.")
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório para salvar resultados.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Número de trials do experimento.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Seed base para randomização.")
    parser.add_argument("--risk-model", type=str, default=RISK_MODEL, choices=["tabicl", "hgb"], help="Modelo para ReCIRC.")
    parser.add_argument("--no-plots", action="store_true", help="Desabilita geração de gráficos.")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--aacrc-repo", type=Path, default=AACRC_REPO)
    parser.add_argument("--aacrc-integration", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--aacrc-ridge", type=float, default=AACRC_RIDGE,
                        help="Penalidade ridge, aplicada apenas às inclinações (nunca ao intercepto).")
    parser.add_argument("--aacrc-maxiter", type=int, default=AACRC_MAXITER)
    parser.add_argument("--aacrc-score", choices=["aps", "rank"], default=AACRC_SCORE,
                        help="Score de conformidade do AA-CRC: 'aps' (contínuo) ou 'rank' (grade discreta, legado).")
    parser.add_argument("--aacrc-fit-split", choices=["risk", "cal"], default=AACRC_FIT_SPLIT,
                        help="Onde ajustar theta: D_risk (default, mesmo orçamento do ReCIRC) ou uma parte de C.")
    parser.add_argument("--aacrc-calib-frac", type=float, default=AACRC_CALIB_FRAC,
                        help="Fração de C reservada ao deslocamento quando --aacrc-fit-split cal.")
    parser.add_argument("--aacrc-no-postfit", action="store_true",
                        help="Desliga a calibração escalar pós-ajuste (reproduz o comportamento sem garantia marginal).")
    parser.add_argument("--aacrc-randomized-offset", action="store_true",
                        help="Randomiza o deslocamento entre pontos de quebra para atingir alpha na igualdade.")
    parser.add_argument("--aacrc-tie", choices=["inclusive", "strict"], default=AACRC_TIE,
                        help="Desempate em score >= u: inclusive (tolerância 1e-9) ou estrito.")
    parser.add_argument("--aacrc-max-fit-rows", type=int, default=None,
                        help="Subamostra o conjunto de ajuste de theta (controle de tempo do SLSQP).")
    parser.add_argument("--data-file", type=Path, default=Path(__file__).resolve().parent / "data_letter_recognition/letter-recognition.csv")
    parser.add_argument("--no-recirc", action="store_true", help="Executa apenas CRC e AA-CRC.")
    parser.add_argument("--self-check", action="store_true", help="Validação offline com NumPy/SciPy.")
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.alpha < 1:
        parser.error("--trials deve ser positivo e --alpha deve estar entre 0 e 1.")
    if not np.isfinite(args.aacrc_ridge) or args.aacrc_ridge < 0 or args.aacrc_maxiter < 1:
        parser.error("--aacrc-ridge deve ser finito e não negativo; --aacrc-maxiter deve ser positivo.")
    if not 0 < args.aacrc_calib_frac < 1:
        parser.error("--aacrc-calib-frac deve estar em (0, 1).")
    if args.aacrc_max_fit_rows is not None and args.aacrc_max_fit_rows < 2:
        parser.error("--aacrc-max-fit-rows deve ser maior que 1.")
    if args.aacrc_randomized_offset and args.aacrc_no_postfit:
        parser.error("--aacrc-randomized-offset exige a calibração pós-ajuste.")
    return args


def main():
    args = parse_args()

    global N_TRIALS, BASE_SEED, RISK_MODEL, OUT_DIR, K, ALPHA
    global AACRC_MODULE, AACRC_OUTPUT_DIR, AACRC_RIDGE, AACRC_MAXITER, AACRC_INTEGRATION
    global AACRC_SCORE, AACRC_FIT_SPLIT, AACRC_CALIB_FRAC, AACRC_POSTFIT
    global AACRC_RANDOMIZED_OFFSET, AACRC_TIE, AACRC_MAX_FIT_ROWS
    ALPHA = args.alpha
    N_TRIALS = args.trials
    BASE_SEED = args.seed
    RISK_MODEL = args.risk_model

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    AACRC_INTEGRATION = args.aacrc_integration
    AACRC_MODULE = load_original_aacrc(args.aacrc_repo, AACRC_INTEGRATION)
    AACRC_OUTPUT_DIR = output_dir
    AACRC_RIDGE, AACRC_MAXITER = args.aacrc_ridge, args.aacrc_maxiter
    AACRC_SCORE = args.aacrc_score
    AACRC_FIT_SPLIT = args.aacrc_fit_split
    AACRC_CALIB_FRAC = args.aacrc_calib_frac
    AACRC_POSTFIT = not args.aacrc_no_postfit
    AACRC_RANDOMIZED_OFFSET = args.aacrc_randomized_offset
    AACRC_TIE = args.aacrc_tie
    AACRC_MAX_FIT_ROWS = args.aacrc_max_fit_rows
    X, y, CLASS_NAMES, raw_df = load_letter_recognition(str(args.data_file))
    N, P = X.shape
    K = len(CLASS_NAMES)
    if K < 2 or int(CAL_FRAC * N) <= 1 / ALPHA:
        raise ValueError("Precisa de ao menos duas classes e len(C) > 1/alpha.")

    print(f"N={N}, p={P}, K={K}")
    print("Classes:", CLASS_NAMES)

    all_summary_rows = []
    all_bin_rows = []
    all_calibration_paths = []

    risk_diagnostic = RiskCalibration()
    for trial in tqdm(range(N_TRIALS), desc="Trials"):
        seed = BASE_SEED + 100 * trial
        summary_rows, bin_rows, paths = run_one_trial(trial=trial, seed=seed, X=X, y=y, K=K, risk_model=RISK_MODEL, use_recirc=not args.no_recirc, diagnostic=risk_diagnostic)
        all_summary_rows.extend(summary_rows)
        all_bin_rows.extend(bin_rows)
        all_calibration_paths.extend(paths)

        pd.DataFrame(all_summary_rows).to_csv(output_dir / "summary_results_incremental.csv", index=False)
        pd.DataFrame(all_bin_rows).to_csv(output_dir / "bin_results_incremental.csv", index=False)

    results_df = pd.DataFrame(all_summary_rows)
    bin_results_df = pd.DataFrame(all_bin_rows)
    if len(all_calibration_paths):
        calibration_paths_df = pd.concat(all_calibration_paths, ignore_index=True)
    else:
        calibration_paths_df = pd.DataFrame()

    risk_diagnostic.save(output_dir, make_plot=not args.no_plots)
    results_df.to_csv(output_dir / "summary_results.csv", index=False)
    bin_results_df.to_csv(output_dir / "bin_results.csv", index=False)
    calibration_paths_df.to_csv(output_dir / "calibration_paths.csv", index=False)

    summary_by_method, bins_by_method, compact_table = build_final_summary(results_df, bin_results_df)
    summary_by_method.to_csv(output_dir / "summary_by_method_all_trials.csv", index=False)
    bins_by_method.to_csv(output_dir / "bins_by_method_all_trials.csv", index=False)
    compact_table.to_csv(output_dir / "compact_summary_all_trials.csv", index=False)

    print("\nResumo por método:")
    print(summary_by_method.round(4).to_string(index=False))
    print("\nTabela compacta:")
    print(compact_table.to_string(index=False))

    if not args.no_plots:
        save_plots(all_summary_rows, all_bin_rows, str(output_dir), alpha=ALPHA)
        print(f"\nGráficos salvos em: {output_dir}")

    meta = {
        "experiment": "letter_recognition_official_aacrc",
        "aacrc": {
            "source_commit": AACRC_COMMIT, "source_sha256": AACRC_SOURCE_SHA256,
            "source_path": str(Path(AACRC_MODULE.__file__).resolve()),
            "objective": "authors J/J_prime", "optimizer": "SLSQP",
            "integration": AACRC_INTEGRATION, "quadrature_nodes": 100,
            "ridge": AACRC_RIDGE, "ridge_on_intercept": False, "maxiter": AACRC_MAXITER,
            "fit_split": "D_risk" if AACRC_FIT_SPLIT == "risk" else f"C[{1 - AACRC_CALIB_FRAC:.2f}]",
            "offset_split": "C" if AACRC_FIT_SPLIT == "risk" else f"C[{AACRC_CALIB_FRAC:.2f}]",
            "feature_standardization_split": "D_risk",
            "sample_correction": "1/n_fit no objetivo; (n+1) no deslocamento",
            "feature_map": "intercept + standardized raw and probability-summary features",
            "threshold_class": "u(x) = Phi(x) @ theta + b",
            "score_mode": AACRC_SCORE,
            "scores": ("infinity for rank 1; 1 - cumulative probability mass otherwise"
                       if AACRC_SCORE == "aps" else "infinity for rank 1; (K-rank+1)/K otherwise"),
            "inclusion": "score >= u", "set_size_range": [1, int(K)],
            "tie_handling": AACRC_TIE,
            "loss": "one-hot false negative = indicator(true_label_rank > m)",
            "postfit_scalar_calibration": bool(AACRC_POSTFIT),
            "randomized_offset": bool(AACRC_RANDOMIZED_OFFSET),
        },
        "data_file": str(args.data_file.resolve()),
        "n_obs": int(N), "n_features": int(P), "class_names": CLASS_NAMES,
        "split_fractions": {"base": BASE_TRAIN_FRAC, "risk": RISK_FRAC, "cal": CAL_FRAC, "test": TEST_FRAC},
        "split_strategy": "stratified by label, inherited from original script",
        "budget_grid": {"construction": "fixed_before_calibration", "values": A_GRID.tolist()},
        "recirc_enabled": not args.no_recirc,
        "alpha": float(ALPHA),
        "n_trials": int(N_TRIALS),
        "seed": int(BASE_SEED),
        "risk_model": str(RISK_MODEL) if not args.no_recirc else None,
        "n_classes": int(K),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
