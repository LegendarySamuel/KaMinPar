/*******************************************************************************
 * @file:   dkaminpar_graphgen.h
 *
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#include "apps/dkaminpar_graphgen.h"

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/parallel.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <tbb/parallel_sort.h>

#include <kagen_interface.h>

namespace dkaminpar::graphgen {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(GeneratorType, generator_type) = {
    {GeneratorType::NONE, "none"},
    {GeneratorType::GNM, "gnm"},
    {GeneratorType::RGG2D, "rgg2d"},
    {GeneratorType::RHG, "rhg"},
};

using namespace kagen::interface;

namespace {
SET_DEBUG(true);

DistributedGraph build_graph(const auto &edge_list, scalable_vector<GlobalNodeID> node_distribution) {
  SCOPED_TIMER("Build graph from edge list");

  const auto [size, rank] = mpi::get_comm_info();
  const GlobalNodeID from = node_distribution[rank];
  const GlobalNodeID to = node_distribution[rank + 1];
  const auto n = static_cast<NodeID>(to - from);
  const auto m = static_cast<EdgeID>(edge_list.size());

  // bucket sort nodes
  START_TIMER("Bucket sort");
  scalable_vector<Atomic<NodeID>> buckets(n);
  tbb::parallel_for<EdgeID>(
      0, m, [&](const EdgeID e) { buckets[edge_list[e].first].fetch_add(1, std::memory_order_relaxed); });
  shm::parallel::prefix_sum(buckets.begin(), buckets.end(), buckets.begin());
  STOP_TIMER();

  // build edges array
  START_TIMER("Build edges array");
  scalable_vector<EdgeID> edges(m);
  graph::GhostNodeMapper ghost_node_mapper(node_distribution);
  tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
    const auto u = static_cast<NodeID>(edge_list[e].first);
    const auto v = static_cast<NodeID>(edge_list[e].second);
    const auto pos = buckets[u].fetch_sub(1, std::memory_order_relaxed) - 1;

    if (v >= from && v < to) {
      edges[pos] = static_cast<NodeID>(v - from);
    } else {
      edges[pos] = ghost_node_mapper.new_ghost_node(v);
    }
  });
  STOP_TIMER();

  auto mapped_ghost_nodes = TIMED_SCOPE("Finalize ghost node mapping") { return ghost_node_mapper.finalize(); };

  // build nodes array
  START_TIMER("Build nodes array");
  scalable_vector<NodeID> nodes(n + 1);
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { nodes[u] = buckets[u]; });
  nodes.back() = m;
  STOP_TIMER();

  DistributedGraph graph{std::move(node_distribution),
                         mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(m, MPI_COMM_WORLD),
                         std::move(nodes),
                         std::move(edges),
                         std::move(mapped_ghost_nodes.ghost_owner),
                         std::move(mapped_ghost_nodes.ghost_to_global),
                         std::move(mapped_ghost_nodes.global_to_ghost),
                         MPI_COMM_WORLD};
  graph::debug::validate(graph);
  return graph;
}

scalable_vector<GlobalNodeID> build_node_distribution(const std::pair<SInt, SInt> range) {
  const auto [size, rank] = mpi::get_comm_info();
  const GlobalNodeID to = range.second + 1;

  scalable_vector<GlobalNodeID> node_distribution(size + 1);
  mpi::allgather(&to, 1, node_distribution.data() + 1, 1);
  return node_distribution;
}
} // namespace

DistributedGraph create_undirected_gmm(const GlobalNodeID n, const GlobalEdgeID m, const BlockID k, const int seed) {
  const auto [edges, range] = TIMED_SCOPE("KaGen") {
    const auto [size, rank] = mpi::get_comm_info();
    return KaGen{rank, size}.GenerateUndirectedGNM(n, m, k, seed);
  };
  return build_graph(edges, build_node_distribution(range));
}

DistributedGraph create_rgg2d(const GlobalNodeID n, const double r, const BlockID k, const int seed) {
  const auto [edges, range] = TIMED_SCOPE("KaGen") {
    const auto [size, rank] = mpi::get_comm_info();
    return KaGen{rank, size}.Generate2DRGG(n, r, k, seed);
  };
  return build_graph(edges, build_node_distribution(range));
}

DistributedGraph create_rhg(const GlobalNodeID n, const double gamma, const NodeID d, const BlockID k, const int seed) {
  const auto [edges, range] = TIMED_SCOPE("KaGen") {
    const auto [size, rank] = mpi::get_comm_info();
    return KaGen{rank, size}.GenerateRHG(n, gamma, d, k, seed);
  };
  return build_graph(edges, build_node_distribution(range));
}

DistributedGraph generate(const GeneratorContext ctx) {
  const int seed = static_cast<int>(shm::Randomize::instance().random_index(0, std::numeric_limits<int>::max()));

  switch (ctx.type) {
  case GeneratorType::NONE:
    FATAL_ERROR << "no graph generator configured";
    break;

  case GeneratorType::GNM:
    return create_undirected_gmm(ctx.n, ctx.m, ctx.k, seed);

  case GeneratorType::RGG2D:
    return create_rgg2d(ctx.n, ctx.r, ctx.k, seed);

  case GeneratorType::RHG:
    return create_rhg(ctx.n, ctx.gamma, ctx.d, ctx.k, seed);
  }

  __builtin_unreachable();
}
} // namespace dkaminpar::graphgen