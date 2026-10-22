//! The learned prior: a logistic model over cheap board features (see
//! [`crate::dfpn::DfpnConfig::prior_weights`]) giving the probability that Black wins.
//! It sets only the initial proof and disproof numbers of new nodes, so it
//! orders the search and never decides a node: any weights give the same
//! results, good weights give them faster.
//!
//! Training data comes from the solver itself: with `--dump-proofs`, a sample
//! of the proofs it stores is written as records of [`RECORD_FLOATS`]
//! little-endian `f32`s (the features, then 1 if Black wins and 0 otherwise,
//! then log2 of the proof's work). Stored proofs are labelled positions: the
//! labels are exact except for the pruned side's losses under `--prune-for`
//! and for path-specific proofs, whose value depends on history the features
//! do not see (noise for training, nothing more). [`fit`] fits the weights by
//! Newton's method.

use crate::dfpn::PRIOR_FEATURES;

/// Floats per dump record: the features, the winner (1 = Black), log2 of the proof's work.
pub const RECORD_FLOATS: usize = PRIOR_FEATURES + 2;

/// The built-in weights: fitted to proof dumps of 3×3, 4×4, 3×7, 3×8, 4×5,
/// 1×10, 1×11, 2×10 and 2×11 solves near each board's value (none of the
/// boards it was then tested on). A second round with four more features and
/// data collected with this model switched on searched no better.
pub const DEFAULT_PRIOR_WEIGHTS: [f32; PRIOR_FEATURES] = [
    2.5991, 3.6981, -1.9082, 1.6500, -6.6835, -5.9863, -6.6522, 10.9024, -10.1402, 7.4062, -3.4231, 2.2141, 4.7098,
    -7.3238,
];

/// The default scale of the initial numbers (`1 + scale · (−ln p)`).
pub const DEFAULT_PRIOR_SCALE: f32 = 64.0;

const F: usize = PRIOR_FEATURES;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

fn dot(weights: &[f64; F], features: &[f32]) -> f64 {
    weights.iter().zip(features).map(|(a, &b)| a * f64::from(b)).sum()
}

/// Solve `m · x = rhs` for a symmetric positive definite `m` (Cholesky).
#[allow(clippy::needless_range_loop)]
fn solve_spd(mut m: [[f64; F]; F], rhs: [f64; F]) -> [f64; F] {
    for col in 0..F {
        let diag = (m[col][col] - (0..col).map(|k| m[col][k] * m[col][k]).sum::<f64>()).max(1e-300).sqrt();
        m[col][col] = diag;
        for row in col + 1..F {
            m[row][col] = (m[row][col] - (0..col).map(|k| m[row][k] * m[col][k]).sum::<f64>()) / diag;
        }
    }
    let mut fwd = [0.0; F];
    for row in 0..F {
        fwd[row] = (rhs[row] - (0..row).map(|k| m[row][k] * fwd[k]).sum::<f64>()) / m[row][row];
    }
    let mut out = [0.0; F];
    for row in (0..F).rev() {
        out[row] = (fwd[row] - (row + 1..F).map(|k| m[k][row] * out[k]).sum::<f64>()) / m[row][row];
    }
    out
}

/// Fit the weights to `datasets` (each a slice of whole records) by Newton's
/// method on the L2-regularised log loss, every dataset weighted equally (so
/// that one long solve does not drown the others).
///
/// # Panics
/// If a dataset is not a whole number of records.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn fit(datasets: &[&[f32]], l2: f64) -> [f32; PRIOR_FEATURES] {
    for set in datasets {
        assert!(set.len() % RECORD_FLOATS == 0, "a dump must hold whole records");
    }
    let sets: Vec<(&[f32], f64)> = datasets
        .iter()
        .filter(|set| !set.is_empty())
        .map(|set| (*set, (RECORD_FLOATS as f64) / set.len() as f64))
        .collect();
    let reg = l2 * sets.len() as f64;
    let mut weights = [0.0f64; F];
    for _ in 0..50 {
        let mut grad = [0.0; F];
        let mut hess = [[0.0; F]; F];
        for &(set, share) in &sets {
            for rec in set.as_chunks::<RECORD_FLOATS>().0 {
                let prob = sigmoid(dot(&weights, &rec[..F]));
                let err = share * (prob - f64::from(rec[F]));
                let curv = share * prob * (1.0 - prob);
                for row in 0..F {
                    let xr = f64::from(rec[row]);
                    grad[row] = err.mul_add(xr, grad[row]);
                    for col in 0..=row {
                        hess[row][col] = (curv * xr).mul_add(f64::from(rec[col]), hess[row][col]);
                    }
                }
            }
        }
        for row in 0..F {
            grad[row] = reg.mul_add(weights[row], grad[row]);
            hess[row][row] += reg;
            for col in 0..row {
                hess[col][row] = hess[row][col];
            }
        }
        let step = solve_spd(hess, grad);
        for (w, st) in weights.iter_mut().zip(step) {
            *w -= st;
        }
        if step.iter().all(|st| st.abs() < 1e-7) {
            break;
        }
    }
    weights.map(|x| x as f32)
}

/// Mean log loss and accuracy of `weights` on `data` (whole records).
#[must_use]
pub fn evaluate(weights: &[f32; PRIOR_FEATURES], data: &[f32]) -> (f64, f64) {
    let wide = weights.map(f64::from);
    let (mut loss, mut right, mut count) = (0.0, 0usize, 0usize);
    for rec in data.as_chunks::<RECORD_FLOATS>().0 {
        let prob = sigmoid(dot(&wide, &rec[..F])).clamp(1e-6, 1.0 - 1e-6);
        let black = rec[F] > 0.5;
        loss -= if black { prob.ln() } else { (1.0 - prob).ln() };
        right += usize::from((prob > 0.5) == black);
        count += 1;
    }
    (loss / count.max(1) as f64, right as f64 / count.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_a_separating_direction() {
        // Labels from a known linear rule on feature 3 (plus the bias): the
        // fitted weight on it must be large and positive, the others small.
        let mut rng = sanko_core::Rng::new(7);
        let mut data = Vec::new();
        for _ in 0..4000 {
            let mut rec = [0.0f32; RECORD_FLOATS];
            rec[0] = 1.0;
            for x in rec.iter_mut().take(F).skip(1) {
                *x = (rng.next_below(1000) as f32) / 1000.0;
            }
            rec[F] = f32::from(u8::from(rec[3] > 0.5));
            data.extend_from_slice(&rec);
        }
        let weights = fit(&[&data], 1e-4);
        assert!(weights[3] > 5.0, "{weights:?}");
        assert!((1..F).filter(|&i| i != 3).all(|i| weights[i].abs() < 1.0), "{weights:?}");
        let (loss, acc) = evaluate(&weights, &data);
        assert!(acc > 0.95 && loss < 0.2, "{loss} {acc}");
    }
}
