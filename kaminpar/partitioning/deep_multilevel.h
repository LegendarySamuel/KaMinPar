/*******************************************************************************
 * @file:   parallel_recursive_bisection.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar/coarsening/lp_clustering.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/factories.h"
#include "kaminpar/graphutils/subgraph_extractor.h"
#include "kaminpar/initial_partitioning/initial_partitioning_facade.h"
#include "kaminpar/initial_partitioning/pool_bipartitioner.h"
#include "kaminpar/partitioning/helper.h"

#include "common/console_io.h"

namespace kaminpar::shm::partitioning {
class DeepMultilevelPartitioner {
  SET_DEBUG(false);
  SET_STATISTICS(false);

public:
  DeepMultilevelPartitioner(const Graph &input_graph, const Context &input_ctx);

  DeepMultilevelPartitioner(const DeepMultilevelPartitioner &) = delete;
  DeepMultilevelPartitioner &operator=(const DeepMultilevelPartitioner &) = delete;
  DeepMultilevelPartitioner(DeepMultilevelPartitioner &&) = delete;
  DeepMultilevelPartitioner &operator=(DeepMultilevelPartitioner &&) = delete;

  PartitionedGraph partition();

private:
  PartitionedGraph uncoarsen(PartitionedGraph p_graph, bool &refined);

  inline PartitionedGraph uncoarsen_once(PartitionedGraph p_graph);

  void refine(PartitionedGraph &p_graph);

  inline void extend_partition(PartitionedGraph &p_graph, BlockID k_prime);

  const Graph *coarsen();

  NodeID initial_partitioning_threshold();

  PartitionedGraph initial_partition(const Graph *graph);

  void print_statistics();

  const Graph &_input_graph;
  const Context &_input_ctx;
  PartitionContext _current_p_ctx;

  // Coarsening
  std::unique_ptr<Coarsener> _coarsener;

  // Refinement
  std::unique_ptr<Refiner> _refiner;

  // Initial partitioning -> subgraph extraction
  graph::SubgraphMemory _subgraph_memory;
  TemporaryGraphExtractionBufferPool _ip_extraction_pool;

  // Initial partitioning
  GlobalInitialPartitionerMemoryPool _ip_m_ctx_pool;
};
} // namespace kaminpar::shm::partitioning
