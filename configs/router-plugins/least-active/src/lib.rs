// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `srtctl-example-least-active`: send each request to the worker with the fewest active requests.
//!
//! The smallest complete router plugin crate: one scorer, one picker, one YAML parameter, and the
//! `register` function Dynamo's frontend calls at startup. Select it from a recipe with
//!
//! ```yaml
//! frontend:
//!   worker_selection:
//!     aggregated:
//!       type: srtctl-example-least-active
//!       parameters:
//!         active_request_weight: 1.0
//! ```

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

/// The policy type name recipes select under `frontend.worker_selection.<role>.type`.
pub const POLICY_TYPE: &str = "srtctl-example-least-active";

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    /// Cost added per active request on a worker.
    #[serde(default = "default_weight")]
    active_request_weight: f64,
}

fn default_weight() -> f64 {
    1.0
}

struct ActiveRequestsScorer {
    weight: f64,
}

impl WorkerScorer for ActiveRequestsScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &self,
        _context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            *cost = self.weight * load.active_requests() as f64;
        }
        Ok(())
    }
}

struct LowestCostPicker;

impl WorkerPicker for LowestCostPicker {
    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    if !parameters.active_request_weight.is_finite() || parameters.active_request_weight < 0.0 {
        return Err(WorkerSelectionPolicyProviderError::new(
            "active_request_weight must be a finite non-negative number",
        ));
    }
    let weight = parameters.active_request_weight;

    Ok(Arc::new(
        move |config: &SelectionConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                vec![Box::new(ActiveRequestsScorer { weight })],
                Box::new(LowestCostPicker),
            )
        },
    ))
}

/// Register this crate's policy types. Dynamo's Python frontend calls this once at startup;
/// add one `register_worker_selection` line per policy type the crate provides.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_once() {
        let mut registry = RouterPluginRegistry::default();
        register(&mut registry).unwrap();
        assert!(matches!(
            register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == POLICY_TYPE
        ));
    }
}
