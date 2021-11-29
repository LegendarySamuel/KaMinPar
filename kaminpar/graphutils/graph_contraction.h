/*******************************************************************************
 * @file:   graph_contraction.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Contracts a clustering and constructs the coarse graph.
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

contraction::Result contract(const Graph &graph,
                             const scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> &clustering,
                             contraction::MemoryContext m_ctx = {});
} // namespace kaminpar::graph