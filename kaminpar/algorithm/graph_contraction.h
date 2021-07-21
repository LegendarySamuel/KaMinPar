/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/datastructure/ts_navigable_linked_list.h"
#include "kaminpar/parallel.h"

namespace kaminpar::graph {
namespace contraction {
struct Edge {
  NodeID target;
  EdgeWeight weight;
};

struct MemoryContext {
  scalable_vector<NodeID> buckets;
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> buckets_index;
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> leader_mapping;
  scalable_vector<NavigationMarker<NodeID, Edge>> all_buffered_nodes;
};

struct Result {
  Graph graph;
  scalable_vector<NodeID> mapping;
  MemoryContext m_ctx;
};
} // namespace contraction

contraction::Result contract(const Graph &r, const scalable_vector<NodeID> &clustering,
                             contraction::MemoryContext m_ctx = {});
} // namespace kaminpar::graph