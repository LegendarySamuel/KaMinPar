/*******************************************************************************
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 * @brief:  Shared-memory JET refiner due to: 
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm {
// Refinement algorithm due to 
// "Jet: Multilevel Graph Partitioning on GPUs"
// by Gilbert et al
class JetRefiner : public Refiner {
public:
  JetRefiner(const Context &ctx);

  JetRefiner(const JetRefiner &) = delete;
  JetRefiner &operator=(const JetRefiner &) = delete;

  JetRefiner(JetRefiner &&) noexcept = default;
  JetRefiner &operator=(JetRefiner &&) = delete;

  void initialize(const PartitionedGraph &) {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx);

private:
  const Context &_ctx;
};
} // namespace kaminpar::shm
