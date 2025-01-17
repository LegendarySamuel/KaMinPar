/*******************************************************************************
 * Utility functions to read/write parts of the partitioner context from/to
 * strings.
 *
 * @file:   context_io.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2022
 ******************************************************************************/
#include "dkaminpar/context_io.h"

#include <iomanip>
#include <ostream>
#include <unordered_map>

#include "dkaminpar/context.h"

#include "common/console_io.h"
#include "common/random.h"

namespace kaminpar::dist {
namespace {
template <typename T> std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
  out << "[";
  bool first = true;
  for (const T &e : vec) {
    if (first) {
      first = false;
    } else {
      out << " -> ";
    }
    out << e;
  }
  return out << "]";
}
} // namespace

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
  return {
      {"multilevel/deep", PartitioningMode::DEEP},
      {"multilevel/kway", PartitioningMode::KWAY},
  };
}

std::ostream &operator<<(std::ostream &out, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return out << "multilevel/deep";
  case PartitioningMode::KWAY:
    return out << "multilevel/kway";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GlobalClusteringAlgorithm> get_global_clustering_algorithms() {
  return {
      {"noop", GlobalClusteringAlgorithm::NOOP},
      {"lp", GlobalClusteringAlgorithm::LP},
      {"hem", GlobalClusteringAlgorithm::HEM},
      {"hem-lp", GlobalClusteringAlgorithm::HEM_LP},
      {"my-lp", GlobalClusteringAlgorithm::MY_LP},
      {"ag-lp", GlobalClusteringAlgorithm::AGLP},
      {"lp2", GlobalClusteringAlgorithm::LP2},
      {"ag-lp2", GlobalClusteringAlgorithm::AGLP2},
      {"mq-lp", GlobalClusteringAlgorithm::MQ_LP},
  };
}

std::ostream &operator<<(std::ostream &out, const GlobalClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case GlobalClusteringAlgorithm::NOOP:
    return out << "noop";
  case GlobalClusteringAlgorithm::LP:
    return out << "lp";
  case GlobalClusteringAlgorithm::HEM:
    return out << "hem";
  case GlobalClusteringAlgorithm::HEM_LP:
    return out << "hem-lp";
  case GlobalClusteringAlgorithm::MY_LP:
    return out << "my-lp";
  case GlobalClusteringAlgorithm::AGLP:
    return out << "ag-lp";
  case GlobalClusteringAlgorithm::LP2:
    return out << "lp2";
  case GlobalClusteringAlgorithm::AGLP2:
    return out << "ag-lp2";
  case GlobalClusteringAlgorithm::MQ_LP:
    return out << "mq-lp";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, LocalClusteringAlgorithm> get_local_clustering_algorithms() {
  return {
      {"noop", LocalClusteringAlgorithm::NOOP},
      {"lp", LocalClusteringAlgorithm::LP},
  };
}

std::ostream &operator<<(std::ostream &out, const LocalClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case LocalClusteringAlgorithm::NOOP:
    return out << "noop";
  case LocalClusteringAlgorithm::LP:
    return out << "lp";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningAlgorithm>
get_initial_partitioning_algorithms() {
  return {
      {"kaminpar", InitialPartitioningAlgorithm::KAMINPAR},
      {"mtkahypar", InitialPartitioningAlgorithm::MTKAHYPAR},
      {"random", InitialPartitioningAlgorithm::RANDOM},
  };
}

std::ostream &operator<<(std::ostream &out, const InitialPartitioningAlgorithm algorithm) {
  switch (algorithm) {
  case InitialPartitioningAlgorithm::KAMINPAR:
    return out << "kaminpar";
  case InitialPartitioningAlgorithm::MTKAHYPAR:
    return out << "mtkahypar";
  case InitialPartitioningAlgorithm::RANDOM:
    return out << "random";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"lp/batches", RefinementAlgorithm::BATCHED_LP},
      {"lp/colors", RefinementAlgorithm::COLORED_LP},
      {"fm/global", RefinementAlgorithm::GLOBAL_FM},
      {"fm/local", RefinementAlgorithm::LOCAL_FM},
      {"greedy-balancer/nodes", RefinementAlgorithm::GREEDY_NODE_BALANCER},
      {"greedy-balancer/clusters", RefinementAlgorithm::GREEDY_CLUSTER_BALANCER},
      {"jet/refiner", RefinementAlgorithm::JET_REFINER},
      {"jet/balancer", RefinementAlgorithm::JET_BALANCER},
  };
}

std::unordered_map<std::string, RefinementAlgorithm> get_balancing_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"greedy-balancer/nodes", RefinementAlgorithm::GREEDY_NODE_BALANCER},
      {"greedy-balancer/clusters", RefinementAlgorithm::GREEDY_CLUSTER_BALANCER},
      {"jet/balancer", RefinementAlgorithm::JET_BALANCER},
  };
};

std::ostream &operator<<(std::ostream &out, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return out << "noop";
  case RefinementAlgorithm::BATCHED_LP:
    return out << "lp/batches";
  case RefinementAlgorithm::COLORED_LP:
    return out << "lp/colors";
  case RefinementAlgorithm::LOCAL_FM:
    return out << "fm/local";
  case RefinementAlgorithm::GLOBAL_FM:
    return out << "fm/global";
  case RefinementAlgorithm::GREEDY_NODE_BALANCER:
    return out << "greedy-balancer/nodes";
  case RefinementAlgorithm::GREEDY_CLUSTER_BALANCER:
    return out << "greedy-balancer/clusters";
  case RefinementAlgorithm::JET_REFINER:
    return out << "jet/refiner";
  case RefinementAlgorithm::JET_BALANCER:
    return out << "jet/balancer";
  }

  return out << "<invalid>";
}

std::string get_refinement_algorithms_description() {
  return std::string(R"(
- noop:                       do nothing
- lp/batches:                 LP where batches are nodes with subsequent IDs
- lp/colors:                  LP where batches are color classes
- fm/local:                   local FM
- fm/global:                  global FM
- jet/refiner:                reimplementation of JET's refinement algorithm)")
             .substr(1) +
         "\n" + get_balancing_algorithms_description();
}

std::string get_balancing_algorithms_description() {
  return std::string(R"(
- jet/balancer:               reimplementation of JET's balancing algorithm
- greedy-balancer/singletons: greedy, move individual nodes
- greedy-balancer/movesets:   greedy, move sets of nodes)")
      .substr(1);
}

std::unordered_map<std::string, LabelPropagationMoveExecutionStrategy>
get_label_propagation_move_execution_strategies() {
  return {
      {"probabilistic", LabelPropagationMoveExecutionStrategy::PROBABILISTIC},
      {"best", LabelPropagationMoveExecutionStrategy::BEST_MOVES},
      {"local", LabelPropagationMoveExecutionStrategy::LOCAL_MOVES},
  };
}

std::ostream &operator<<(std::ostream &out, const LabelPropagationMoveExecutionStrategy strategy) {
  switch (strategy) {
  case LabelPropagationMoveExecutionStrategy::PROBABILISTIC:
    return out << "probabilistic";
  case LabelPropagationMoveExecutionStrategy::BEST_MOVES:
    return out << "best";
  case LabelPropagationMoveExecutionStrategy::LOCAL_MOVES:
    return out << "local";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GraphOrdering> get_graph_orderings() {
  return {
      {"natural", GraphOrdering::NATURAL},
      {"deg-buckets", GraphOrdering::DEGREE_BUCKETS},
      {"degree-buckets", GraphOrdering::DEGREE_BUCKETS},
      {"coloring", GraphOrdering::COLORING},
  };
}

std::ostream &operator<<(std::ostream &out, const GraphOrdering ordering) {
  switch (ordering) {
  case GraphOrdering::NATURAL:
    return out << "natural";
  case GraphOrdering::DEGREE_BUCKETS:
    return out << "deg-buckets";
  case GraphOrdering::COLORING:
    return out << "coloring";
  }

  return out << "<invalid>";
}

std::ostream &operator<<(std::ostream &out, const ClusterSizeStrategy strategy) {
  switch (strategy) {
  case ClusterSizeStrategy::ZERO:
    return out << "zero";
  case ClusterSizeStrategy::ONE:
    return out << "one";
  case ClusterSizeStrategy::MAX_OVERLOAD:
    return out << "max-overload";
  case ClusterSizeStrategy::AVG_OVERLOAD:
    return out << "avg-overload";
  case ClusterSizeStrategy::MIN_OVERLOAD:
    return out << "min-overload";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterSizeStrategy> get_move_set_size_strategies() {
  return {
      {"zero", ClusterSizeStrategy::ZERO},
      {"one", ClusterSizeStrategy::ONE},
      {"max-overload", ClusterSizeStrategy::MAX_OVERLOAD},
      {"avg-overload", ClusterSizeStrategy::AVG_OVERLOAD},
      {"min-overload", ClusterSizeStrategy::MIN_OVERLOAD},
  };
}

std::ostream &operator<<(std::ostream &out, const ClusterStrategy strategy) {
  switch (strategy) {
  case ClusterStrategy::SINGLETONS:
    return out << "singletons";
  case ClusterStrategy::LP:
    return out << "lp";
  case ClusterStrategy::GREEDY_BATCH_PREFIX:
    return out << "greedy-batch-prefix";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterStrategy> get_move_set_strategies() {
  return {
      {"singletons", ClusterStrategy::SINGLETONS},
      {"lp", ClusterStrategy::LP},
      {"greedy-batch-prefix", ClusterStrategy::GREEDY_BATCH_PREFIX},
  };
}

void print(const Context &ctx, const bool root, std::ostream &out, MPI_Comm comm) {
  if (root) {
    out << "Seed:                         " << Random::seed << "\n";
    out << "Graph:\n";
    out << "  Rearrange graph by:         " << ctx.rearrange_by << "\n";
  }
  print(ctx.partition, root, out, comm);
  if (root) {
    cio::print_delimiter("Partitioning Scheme", '-');

    out << "Partitioning mode:            " << ctx.mode << "\n";
    if (ctx.mode == PartitioningMode::DEEP) {
      out << "  Enable PE-splitting:        " << (ctx.enable_pe_splitting ? "yes" : "no") << "\n";
      out << "  Partition extension factor: " << ctx.partition.K << "\n";
      out << "  Simulate seq. hybrid exe.:  " << (ctx.simulate_singlethread ? "yes" : "no") << "\n";
    }
    cio::print_delimiter("Coarsening", '-');
    print(ctx.coarsening, ctx.parallel, out);
    cio::print_delimiter("Initial Partitioning", '-');
    print(ctx.initial_partitioning, out);
    cio::print_delimiter("Refinement", '-');
    print(ctx.refinement, out);
  }
}

void print(const PartitionContext &ctx, const bool root, std::ostream &out, MPI_Comm comm) {
  // If the graph context has not been initialized with a graph, be silent
  // (This should never happen)
  if (ctx.graph == nullptr) {
    return;
  }

  const auto size = std::max<std::uint64_t>({
      static_cast<std::uint64_t>(ctx.graph->global_n),
      static_cast<std::uint64_t>(ctx.graph->global_m),
      static_cast<std::uint64_t>(ctx.graph->max_block_weight(0)),
  });
  const auto width = std::ceil(std::log10(size)) + 1;

  const GlobalNodeID num_global_total_nodes =
      mpi::allreduce<GlobalNodeID>(ctx.graph->total_n, MPI_SUM, comm);

  if (root) {
    out << "  Number of global nodes:    " << std::setw(width) << ctx.graph->global_n;
    if (asserting_cast<GlobalNodeWeight>(ctx.graph->global_n) ==
        ctx.graph->global_total_node_weight) {
      out << " (unweighted)\n";
    } else {
      out << " (total weight: " << ctx.graph->global_total_node_weight << ")\n";
    }
    out << "    + ghost nodes:           " << std::setw(width)
        << num_global_total_nodes - ctx.graph->global_n << "\n";
    out << "  Number of global edges:    " << std::setw(width) << ctx.graph->global_m;
    if (asserting_cast<GlobalEdgeWeight>(ctx.graph->global_m) ==
        ctx.graph->global_total_edge_weight) {
      out << " (unweighted)\n";
    } else {
      out << " (total weight: " << ctx.graph->global_total_edge_weight << ")\n";
    }
    out << "Number of blocks:             " << ctx.k << "\n";
    out << "Maximum block weight:         " << ctx.graph->max_block_weight(0) << " ("
        << ctx.graph->perfectly_balanced_block_weight(0) << " + " << 100 * ctx.epsilon << "%)\n";
  }
}

void print(const CoarseningContext &ctx, const ParallelContext &parallel, std::ostream &out) {
  out << "Contraction limit:            " << ctx.contraction_limit << "\n";
  if (ctx.max_global_clustering_levels > 0 && ctx.max_local_clustering_levels > 0) {
    out << "Coarsening mode:              local[" << ctx.max_local_clustering_levels << "]+global["
        << ctx.max_global_clustering_levels << "]\n";
  } else if (ctx.max_global_clustering_levels > 0) {
    out << "Coarsening mode:              global[" << ctx.max_global_clustering_levels << "]\n";
  } else if (ctx.max_local_clustering_levels > 0) {
    out << "Coarsening mode:              local[" << ctx.max_local_clustering_levels << "]\n";
  } else {
    out << "Coarsening mode:              disabled\n";
  }

  if (ctx.max_local_clustering_levels > 0) {
    out << "Local clustering algorithm:   " << ctx.local_clustering_algorithm << "\n";
    out << "  Number of iterations:       " << ctx.local_lp.num_iterations << "\n";
    out << "  High degree threshold:      " << ctx.local_lp.passive_high_degree_threshold
        << " (passive), " << ctx.local_lp.active_high_degree_threshold << " (active)\n";
    out << "  Max degree:                 " << ctx.local_lp.max_num_neighbors << "\n";
    out << "  Ghost nodes:                "
        << (ctx.local_lp.ignore_ghost_nodes ? "ignore" : "consider") << "+"
        << (ctx.local_lp.keep_ghost_clusters ? "keep" : "discard") << "\n";
  }

  if (ctx.max_global_clustering_levels > 0) {
    out << "Global clustering algorithm:  " << ctx.global_clustering_algorithm;
    if (ctx.max_cnode_imbalance < std::numeric_limits<double>::max()) {
      out << " [rebalance if >" << std::setprecision(2) << 100.0 * (ctx.max_cnode_imbalance - 1.0)
          << "%";
      if (ctx.migrate_cnode_prefix) {
        out << ", prefix";
      } else {
        out << ", suffix";
      }
      if (ctx.force_perfect_cnode_balance) {
        out << ", strict";
      } else {
        out << ", relaxed";
      }
      out << "]";
    } else {
      out << "[natural assignment]";
    }
    out << "\n";

    if (ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::LP ||
        ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::HEM_LP) {
      out << "  Number of iterations:       " << ctx.global_lp.num_iterations << "\n";
      out << "  High degree threshold:      " << ctx.global_lp.passive_high_degree_threshold
          << " (passive), " << ctx.global_lp.active_high_degree_threshold << " (active)\n";
      out << "  Max degree:                 " << ctx.global_lp.max_num_neighbors << "\n";
      if (ctx.global_lp.fixed_num_chunks == 0) {
        out << "  Number of chunks:           " << ctx.global_lp.compute_num_chunks(parallel)
            << "[= max(" << ctx.global_lp.min_num_chunks << ", " << ctx.global_lp.total_num_chunks
            << " / " << parallel.num_mpis
            << (ctx.global_lp.scale_chunks_with_threads
                    ? std::string(" / ") + std::to_string(parallel.num_threads)
                    : "")
            << "]\n";
      } else {
        out << "  Number of chunks:           " << ctx.global_lp.fixed_num_chunks << "\n";
      }
      // out << "  Number of chunks:           " << ctx.global_lp.num_chunks
      //<< " (min: " << ctx.global_lp.min_num_chunks << ", total: " <<
      // ctx.global_lp.total_num_chunks << ")"
      //<< (ctx.global_lp.scale_chunks_with_threads ? ", scaled" : "") << "\n";
      out << "  Active set:                 "
          << (ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::LP ? "no" : "yes")
          << "\n";
      out << "  Cluster weights:            "
          << (ctx.global_lp.sync_cluster_weights ? "sync" : "no-sync") << "+"
          << (ctx.global_lp.enforce_cluster_weights ? "enforce" : "no-enforce") << " "
          << (ctx.global_lp.cheap_toplevel ? "(on level > 1)" : "(always)") << "\n";
    }

    if (ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::HEM ||
        ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::HEM_LP) {
      // out << "  Number of coloring ssteps:  " << ctx.hem.num_coloring_chunks
      //<< " (min: " << ctx.hem.min_num_coloring_chunks << ", max: " <<
      // ctx.hem.max_num_coloring_chunks << ")"
      //<< (ctx.hem.scale_coloring_chunks_with_threads ? ", scaled with threads"
      //: "") << "\n";
      out << "  Small color blacklist:      " << 100 * ctx.hem.small_color_blacklist << "%"
          << (ctx.hem.only_blacklist_input_level ? " (input level only)" : "") << "\n";
    }
  }
}

void print(const InitialPartitioningContext &ctx, std::ostream &out) {
  out << "IP algorithm:                 " << ctx.algorithm << "\n";
  if (ctx.algorithm == InitialPartitioningAlgorithm::KAMINPAR) {
    out << "  Configuration preset:       default\n";
  }
}

void print(const RefinementContext &ctx, std::ostream &out) {
  out << "Refinement algorithms:        " << ctx.algorithms << "\n";
  out << "Refine initial partition:     " << (ctx.refine_coarsest_level ? "yes" : "no") << "\n";
  if (ctx.includes_algorithm(RefinementAlgorithm::BATCHED_LP)) {
    out << "Label propagation:\n";
    out << "  Number of iterations:       " << ctx.lp.num_iterations << "\n";
    // out << "  Number of chunks:           " << ctx.lp.num_chunks << " (min: "
    // << ctx.lp.min_num_chunks
    //<< ", total: " << ctx.lp.total_num_chunks << ")" <<
    //(ctx.lp.scale_chunks_with_threads ? ", scaled" : "")
    //<< "\n";
    out << "  Use probabilistic moves:    " << (ctx.lp.ignore_probabilities ? "no" : "yes") << "\n";
    out << "  Number of retries:          " << ctx.lp.num_move_attempts << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::COLORED_LP)) {
    out << "Colored Label Propagation:\n";
    // out << "  Number of coloring ssteps:  " <<
    // ctx.colored_lp.num_coloring_chunks
    //<< " (min: " << ctx.colored_lp.min_num_coloring_chunks
    //<< ", max: " << ctx.colored_lp.max_num_coloring_chunks << ")"
    //<< (ctx.colored_lp.scale_coloring_chunks_with_threads ? ", scaled with
    // threads" : "") << "\n";
    out << "  Number of iterations:       " << ctx.colored_lp.num_iterations << "\n";
    out << "  Commitment strategy:        " << ctx.colored_lp.move_execution_strategy << "\n";
    if (ctx.colored_lp.move_execution_strategy ==
        LabelPropagationMoveExecutionStrategy::PROBABILISTIC) {
      out << "    Number of attempts:       " << ctx.colored_lp.num_probabilistic_move_attempts
          << "\n";
    } else if (ctx.colored_lp.move_execution_strategy == LabelPropagationMoveExecutionStrategy::BEST_MOVES) {
      out << "    Sort by:                  "
          << (ctx.colored_lp.sort_by_rel_gain ? "relative gain" : "absolute gain") << "\n";
    }
    out << "  Commitment rounds:          " << ctx.colored_lp.num_move_execution_iterations << "\n";
    out << "  Track block weights:        "
        << (ctx.colored_lp.track_local_block_weights ? "yes" : "no") << "\n";
    out << "  Use active set:             " << (ctx.colored_lp.use_active_set ? "yes" : "no")
        << "\n";
    out << "  Small color blacklist:      " << 100 * ctx.colored_lp.small_color_blacklist << "%"
        << (ctx.colored_lp.only_blacklist_input_level ? " (input level only)" : "") << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER)) {
    out << "Jet refinement:\n";
    out << "  Number of iterations:       " << ctx.jet.num_iterations << "\n";
    out << "  C:                          [" << ctx.jet.min_c << ".." << ctx.jet.max_c << "] "
        << (ctx.jet.interpolate_c ? "interpolate" : "switch") << "\n";
    out << "  Abortion threshold          "
        << (ctx.jet.use_abortion_threshold ? std::to_string(ctx.jet.abortion_threshold) : "disabled"
           )
        << "\n";
    out << "  Balancing algorithm:        " << ctx.jet.balancing_algorithm << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::GLOBAL_FM)) {
    out << "Global FM refinement:\n";
    out << "  Number of iterations:       " << ctx.fm.num_global_iterations << " x "
        << ctx.fm.num_local_iterations << "\n";
    out << "  Search radius:              " << ctx.fm.max_radius << " via " << ctx.fm.max_hops
        << " hop(s)\n";
    out << "  Revert batch-local moves:   "
        << (ctx.fm.revert_local_moves_after_batch ? "yes" : "no") << "\n";
    out << "  Rollback to best partition: " << (ctx.fm.rollback_deterioration ? "yes" : "no")
        << "\n";
    out << "  Rebalance algorithm:        " << ctx.fm.balancing_algorithm << "\n";
    out << "    Rebalance after iter.:    "
        << (ctx.fm.rebalance_after_each_global_iteration ? "yes" : "no") << "\n";
    out << "    Rebalance after ref.:     " << (ctx.fm.rebalance_after_refinement ? "yes" : "no")
        << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::GREEDY_NODE_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER) &&
       ctx.jet.balancing_algorithm == RefinementAlgorithm::GREEDY_NODE_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::GLOBAL_FM) &&
       ctx.fm.balancing_algorithm == RefinementAlgorithm::GREEDY_NODE_BALANCER)) {
    out << "Greedy balancer:\n";
    out << "  Number of nodes per block:  " << ctx.greedy_balancer.num_nodes_per_block << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::GREEDY_CLUSTER_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER) &&
       ctx.jet.balancing_algorithm == RefinementAlgorithm::GREEDY_CLUSTER_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::GLOBAL_FM) &&
       ctx.fm.balancing_algorithm == RefinementAlgorithm::GREEDY_CLUSTER_BALANCER)) {
    out << "Greedy cluster balancer:\n";
    out << "  Clusters:                   " << ctx.cluster_balancer.cluster_strategy << "\n";
    out << "    Max weight:               " << ctx.cluster_balancer.cluster_size_strategy << " x "
        << ctx.cluster_balancer.cluster_size_multiplier << "\n";
    out << "    Rebuild interval:         "
        << (ctx.cluster_balancer.cluster_rebuild_interval == 0
                ? "never"
                : std::string("every ") +
                      std::to_string(ctx.cluster_balancer.cluster_rebuild_interval) + " round(s)")
        << "\n";
    out << "  Maximum number of rounds:   " << ctx.cluster_balancer.max_num_rounds << "\n";
    out << "  Sequential balancing:       "
        << (ctx.cluster_balancer.enable_sequential_balancing ? "enabled" : "disabled") << "\n";
    out << "    No. of nodes per block:   " << ctx.cluster_balancer.seq_num_nodes_per_block << "\n";
    out << "    Keep all nodes in PQ:     " << (ctx.cluster_balancer.seq_full_pq ? "yes" : "no")
        << "\n";
    out << "  Parallel balancing:         "
        << (ctx.cluster_balancer.enable_parallel_balancing ? "enabled" : "disabled") << "\n";
    out << "    Trigger threshold:        " << ctx.cluster_balancer.parallel_threshold << "\n";
    out << "    # of dicing attempts:     " << ctx.cluster_balancer.par_num_dicing_attempts
        << " --> " << (ctx.cluster_balancer.par_accept_imbalanced ? "accept" : "reject") << "\n";
    out << "    Gain buckets:             log" << ctx.cluster_balancer.par_gain_bucket_factor
        << ", positive gain buckets: "
        << (ctx.cluster_balancer.par_use_positive_gain_buckets ? "yes" : "no") << "\n";
    out << "    Parallel rebalancing:     start at "
        << 100.0 * ctx.cluster_balancer.par_initial_rebalance_fraction << "%, increase by "
        << 100.0 * ctx.cluster_balancer.par_rebalance_fraction_increase << "% each round\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::JET_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER) &&
       ctx.jet.balancing_algorithm == RefinementAlgorithm::JET_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::GLOBAL_FM) &&
       ctx.fm.balancing_algorithm == RefinementAlgorithm::JET_BALANCER)) {
    out << "Jet balancer:\n";
    out << "  Number of iterations:       " << ctx.jet_balancer.num_weak_iterations << " weak + "
        << ctx.jet_balancer.num_strong_iterations << " strong\n";
  }
}
} // namespace kaminpar::dist
