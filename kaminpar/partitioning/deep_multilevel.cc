/*******************************************************************************
 * @file:   parallel_recursive_bisection.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning/deep_multilevel.h"

#include "kaminpar/debug.h"
#include "kaminpar/partitioning/async_initial_partitioning.h"
#include "kaminpar/partitioning/sync_initial_partitioning.h"

namespace kaminpar::shm::partitioning {
DeepMultilevelPartitioner::DeepMultilevelPartitioner(
    const Graph &input_graph, const Context &input_ctx
)
    : _input_graph{input_graph},
      _input_ctx{input_ctx},
      _current_p_ctx{input_ctx.partition}, //
      _coarsener{factory::create_coarsener(input_graph, input_ctx.coarsening)},
      _refiner{factory::create_refiner(input_ctx)},
      _subgraph_memory{input_graph.n(), input_ctx.partition.k, input_graph.m(), true, true} {}

PartitionedGraph DeepMultilevelPartitioner::partition() {
  cio::print_delimiter("Partitioning");

  const Graph *c_graph = coarsen();

  PartitionedGraph p_graph = initial_partition(c_graph);

  bool refined;
  p_graph = uncoarsen(std::move(p_graph), refined);
  if (!refined) {
    refine(p_graph);
  }
  if (p_graph.k() < _input_ctx.partition.k) {
    extend_partition(p_graph, _input_ctx.partition.k);
    refine(p_graph);
  }

  if constexpr (kStatistics) {
    print_statistics();
  }

  return p_graph;
}

PartitionedGraph DeepMultilevelPartitioner::uncoarsen_once(PartitionedGraph p_graph) {
  return helper::uncoarsen_once(_coarsener.get(), std::move(p_graph), _current_p_ctx);
}

void DeepMultilevelPartitioner::refine(PartitionedGraph &p_graph) {
  // If requested, dump the current partition to disk before refinement ...
  debug::dump_partition_hierarchy(p_graph, _coarsener->size(), "pre-refinement", _input_ctx.debug);

  LOG << "  Running refinement on " << p_graph.k() << " blocks";
  helper::refine(_refiner.get(), p_graph, _current_p_ctx);
  LOG << "    Cut:       " << metrics::edge_cut(p_graph);
  LOG << "    Imbalance: " << metrics::imbalance(p_graph);
  LOG << "    Feasible:  " << metrics::is_feasible(p_graph, _current_p_ctx);

  // ... and dump it after refinement.
  debug::dump_partition_hierarchy(p_graph, _coarsener->size(), "post-refinement", _input_ctx.debug);
}

void DeepMultilevelPartitioner::extend_partition(PartitionedGraph &p_graph, const BlockID k_prime) {
  LOG << "  Extending partition from " << p_graph.k() << " blocks to " << k_prime << " blocks";
  helper::extend_partition(
      p_graph,
      k_prime,
      _input_ctx,
      _current_p_ctx,
      _subgraph_memory,
      _ip_extraction_pool,
      _ip_m_ctx_pool
  );
  LOG << "    Cut:       " << metrics::edge_cut(p_graph);
  LOG << "    Imbalance: " << metrics::imbalance(p_graph);
}

PartitionedGraph DeepMultilevelPartitioner::uncoarsen(PartitionedGraph p_graph, bool &refined) {
  LOG << "Uncoarsening -> Level " << _coarsener.get()->size();

  while (!_coarsener->empty()) {
    p_graph = uncoarsen_once(std::move(p_graph));
    refine(p_graph);
    refined = true;

    const BlockID desired_k = helper::compute_k_for_n(p_graph.n(), _input_ctx);
    if (p_graph.k() < desired_k) {
      extend_partition(p_graph, desired_k);
      refined = false;
    }
  }

  return p_graph;
}

const Graph *DeepMultilevelPartitioner::coarsen() {
  const Graph *c_graph = &_input_graph;
  bool shrunk = true;

  while (shrunk && c_graph->n() > initial_partitioning_threshold()) {
    // If requested, dump graph before each coarsening step + after coarsening
    // converged. This way, we also have a dump of the (reordered) input graph,
    // which makes it easier to use the final partition (before reordering it).
    // We dump the coarsest graph in ::initial_partitioning().
    debug::dump_graph_hierarchy(*c_graph, _coarsener->size(), _input_ctx.debug);

    // Build next coarse graph
    shrunk = helper::coarsen_once(_coarsener.get(), c_graph, _input_ctx, _current_p_ctx);
    c_graph = _coarsener->coarsest_graph();

    // Print some metrics for the coarse graphs
    const NodeWeight max_cluster_weight =
        compute_max_cluster_weight(*c_graph, _input_ctx.partition, _input_ctx.coarsening);
    LOG << "Coarsening -> Level " << _coarsener.get()->size();
    LOG << "  Number of nodes: " << c_graph->n() << " | Number of edges: " << c_graph->m();
    LOG << "  Maximum node weight: " << c_graph->max_node_weight() << " <= " << max_cluster_weight;
    LOG;
  }

  if (shrunk) {
    LOG << "==> Coarsening terminated with less than " << initial_partitioning_threshold()
        << " nodes.";
    LOG;
  } else {
    LOG << "==> Coarsening converged.";
    LOG;
  }

  return c_graph;
}

NodeID DeepMultilevelPartitioner::initial_partitioning_threshold() {
  if (helper::parallel_ip_mode(_input_ctx.initial_partitioning.mode)) {
    return _input_ctx.parallel.num_threads * _input_ctx.coarsening.contraction_limit; // p * C
  } else {
    return 2 * _input_ctx.coarsening.contraction_limit; // 2 * C
  }
}

PartitionedGraph DeepMultilevelPartitioner::initial_partition(const Graph *graph) {
  SCOPED_TIMER("Initial partitioning scheme");
  LOG << "Initial partitioning:";

  // If requested, dump the coarsest graph to disk. Note that in the context of
  // deep multilevel, this is not actually the coarsest graph, but rather the
  // coarsest graph before splitting PEs and duplicating the graph.
  // Disable worker splitting with --i-mode=sequential to obtain coarser graphs.
  debug::dump_coarsest_graph(*graph, _input_ctx.debug);
  debug::dump_graph_hierarchy(*graph, _coarsener->size(), _input_ctx.debug);

  // Since timers are not multi-threaded, we disable them during (parallel)
  // initial partitioning.
  DISABLE_TIMERS();
  PartitionedGraph p_graph = [&] {
    switch (_input_ctx.initial_partitioning.mode) {
    case InitialPartitioningMode::SEQUENTIAL:
      return helper::bipartition(graph, _input_ctx.partition.k, _input_ctx, _ip_m_ctx_pool);

    case InitialPartitioningMode::SYNCHRONOUS_PARALLEL:
      return SyncInitialPartitioner(_input_ctx, _ip_m_ctx_pool, _ip_extraction_pool)
          .partition(_coarsener.get(), _current_p_ctx);

    case InitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
      return AsyncInitialPartitioner(_input_ctx, _ip_m_ctx_pool, _ip_extraction_pool)
          .partition(_coarsener.get(), _current_p_ctx);
    }

    __builtin_unreachable();
  }();
  ENABLE_TIMERS();
  helper::update_partition_context(_current_p_ctx, p_graph);

  // Print some metrics for the initial partition.
  LOG << "  Number of blocks: " << p_graph.k();
  LOG << "  Cut:              " << metrics::edge_cut(p_graph);
  LOG << "  Imbalance:        " << metrics::imbalance(p_graph);
  LOG << "  Feasible:         " << (metrics::is_feasible(p_graph, _current_p_ctx) ? "yes" : "no");
  LOG;

  // If requested, dump the coarsest partition -- as noted above, this is not
  // actually the coarsest partition when using deep multilevel.
  debug::dump_coarsest_partition(p_graph, _input_ctx.debug);
  debug::dump_partition_hierarchy(p_graph, _coarsener->size(), "post-refinement", _input_ctx.debug);

  return p_graph;
}

void DeepMultilevelPartitioner::print_statistics() {
  std::size_t num_ip_m_ctx_objects = 0;
  std::size_t max_ip_m_ctx_objects = 0;
  std::size_t min_ip_m_ctx_objects = std::numeric_limits<std::size_t>::max();
  std::size_t ip_m_ctx_memory_in_kb = 0;
  for (const auto &pool : _ip_m_ctx_pool) {
    num_ip_m_ctx_objects += pool.pool.size();
    max_ip_m_ctx_objects = std::max(max_ip_m_ctx_objects, pool.pool.size());
    min_ip_m_ctx_objects = std::min(min_ip_m_ctx_objects, pool.pool.size());
    ip_m_ctx_memory_in_kb += pool.memory_in_kb();
  }

  LOG << logger::CYAN << "Initial partitioning: Memory pool";
  LOG << logger::CYAN << " * # of pool objects: " << min_ip_m_ctx_objects
      << " <= " << 1.0 * num_ip_m_ctx_objects / _input_ctx.parallel.num_threads
      << " <= " << max_ip_m_ctx_objects;
  LOG << logger::CYAN << " * total memory: " << ip_m_ctx_memory_in_kb / 1000 << " Mb";

  std::size_t extraction_nodes_reallocs = 0;
  std::size_t extraction_edges_reallocs = 0;
  std::size_t extraction_memory_in_kb = 0;
  for (const auto &buffer : _ip_extraction_pool) {
    extraction_nodes_reallocs += buffer.num_node_reallocs;
    extraction_edges_reallocs += buffer.num_edge_reallocs;
    extraction_memory_in_kb += buffer.memory_in_kb();
  }

  LOG << logger::CYAN << "Extraction buffer pool:";
  LOG << logger::CYAN << " * # of node buffer reallocs: " << extraction_nodes_reallocs
      << ", # of edge buffer reallocs: " << extraction_edges_reallocs;
  LOG << logger::CYAN << " * total memory: " << extraction_memory_in_kb / 1000 << " Mb";
}
} // namespace kaminpar::shm::partitioning
