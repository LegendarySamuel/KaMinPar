/*******************************************************************************
 * @file:   global_clustering_contraction_redistribution.cc
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Shared-memory parallel contraction of global clustering without
 * any restrictions.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_clustering_contraction.h"

#include <oneapi/tbb/task_arena.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/task_arena.h>

#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi/graph_communication.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/noinit_vector.h"
#include "common/parallel/atomic.h"
#include "common/parallel/loops.h"
#include "common/parallel/vector_ets.h"

namespace kaminpar::dist {
using namespace helper;

namespace {
SET_DEBUG(false);

/*!
 * Sparse all-to-all to exchange coarse node IDs of ghost nodes.
 * @param graph
 * @param label_mapping Current coarse node IDs, must be of size \code{graph.total_n()}, i.e., large enough to store
 * coarse node IDs of owned nodes and ghost nodes.
 */
template <typename LabelMapping>
void exchange_ghost_node_mapping(const DistributedGraph& graph, LabelMapping& label_mapping) {
    SCOPED_TIMER("Exchange ghost node mapping", TIMER_DETAIL);

    struct Message {
        NodeID       local_node;
        GlobalNodeID coarse_global_node;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<Message>(
        graph,
        [&](const NodeID u) -> Message {
            return {u, label_mapping[u]};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_pe, coarse_global_node] = buffer[i];
                const GlobalNodeID global_node                     = graph.offset_n(pe) + local_node_on_pe;
                const auto         local_node                      = graph.global_to_local_node(global_node);

                label_mapping[local_node] = coarse_global_node;
            });
        }
    );
}

using UsedClustersMap    = tbb::concurrent_hash_map<NodeID, NodeID>;
using UsedClustersVector = scalable_vector<NodeID>;

/*!
 * Given a graph with a mapping from nodes to clusters, finds the unique set of clusters that are used by the mapped
 * nodes. Each cluster is owned by some PE (determined by \c resolve_cluster_callback). For each PE, the function
 * returns a map and a vector of local cluster IDs used by the mapped nodes of this PE.
 *
 * @tparam ResolveClusterCallback
 * @param graph
 * @param clustering
 * @param resolve_cluster_callback Given a cluster ID, returns the owner PE (PEID) and the local node/cluster ID
 * (NodeID).
 * @return First component: for each PE \c p, a map mapping local cluster IDs on PE \c p used by mapped nodes on this
 * PE to entries in the second component; Second component: for each PE \c p, a vector containing all local cluster IDs
 * on PE \c p used by mapped nodes on this PE.
 */
template <typename ResolveClusterCallback, typename Clustering>
std::pair<std::vector<UsedClustersMap>, std::vector<UsedClustersVector>> find_used_cluster_ids_per_pe(
    const DistributedGraph& graph, const Clustering& clustering, ResolveClusterCallback&& resolve_cluster_callback
) {
    SCOPED_TIMER("Find used cluster IDs per PE", TIMER_DETAIL);

    const auto size = mpi::get_comm_size(graph.communicator());

    // mark global node IDs that are used as cluster IDs
    std::vector<UsedClustersMap>          used_clusters_map(size);
    std::vector<parallel::Atomic<NodeID>> next_slot_for_pe(size);

    graph.pfor_nodes_range([&](const auto r) {
        tbb::concurrent_hash_map<NodeID, NodeID>::accessor accessor;

        for (NodeID u = r.begin(); u != r.end(); ++u) {
            const GlobalNodeID u_cluster                  = clustering[u];
            const auto [u_cluster_owner, u_local_cluster] = resolve_cluster_callback(u_cluster);

            if (used_clusters_map[u_cluster_owner].insert(accessor, u_local_cluster)) {
                accessor->second = next_slot_for_pe[u_cluster_owner]++;
            }
        }
    });

    // used_clusters_vec[pe] holds local node IDs of PE pe that are used as cluster IDs on this PE
    std::vector<UsedClustersVector> used_clusters_vec(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        used_clusters_vec[pe].resize(used_clusters_map[pe].size());
        tbb::parallel_for(used_clusters_map[pe].range(), [&](const auto r) {
            for (auto it = r.begin(); it != r.end(); ++it) {
                used_clusters_vec[pe][it->second] = it->first;
            }
        });
    });

    return {std::move(used_clusters_map), std::move(used_clusters_vec)};
}

// global mapping, global number of coarse nodes
struct MappingResult {
    GlobalMapping                 mapping;
    scalable_vector<GlobalNodeID> distribution;
};

/*!
 * Compute a label mapping from fine nodes to coarse nodes.
 * @param graph The distributed graph.
 * @param clustering The global clustering to be contracted.
 * @return Label mapping and coarse node distribution. The coarse node distribution is such that coarse nodes are placed
 * on the PE which owned the corresponding cluster ID, i.e., if cluster ID \c x is owned by PE \c y, the coarse node ID
 * \c x is mapped to is also owned by PE \c y.
 */
MappingResult compute_mapping(
    const DistributedGraph& graph, const scalable_vector<parallel::Atomic<GlobalNodeID>>& clustering,
    const bool migrate_nodes = false
) {
    SCOPED_TIMER("Compute coarse node mapping", TIMER_DETAIL);

    const auto size = mpi::get_comm_size(graph.communicator());
    const auto rank = mpi::get_comm_rank(graph.communicator());

    auto used_clusters = find_used_cluster_ids_per_pe(graph, clustering, [&](const GlobalNodeID cluster) {
        if (graph.is_owned_global_node(cluster)) {
            return std::make_pair(rank, graph.global_to_local_node(cluster));
        } else {
            const PEID owner = graph.find_owner_of_global_node(cluster);
            const auto local = static_cast<NodeID>(cluster - graph.offset_n(owner));
            return std::make_pair(owner, local);
        }
    });

    auto& used_clusters_map = used_clusters.first;
    auto& used_clusters_vec = used_clusters.second;

    // send each PE its local node IDs that are used as cluster IDs somewhere
    const auto in_msg = mpi::sparse_alltoall_get<NodeID>(std::move(used_clusters_vec), graph.communicator());

    // map local labels to consecutive coarse node IDs
    scalable_vector<parallel::Atomic<GlobalNodeID>> label_mapping(graph.total_n());
    parallel::chunked_for(in_msg, [&](const NodeID local_label) {
        KASSERT(local_label < graph.n());
        label_mapping[local_label].store(1, std::memory_order_relaxed);
    });
    parallel::prefix_sum(label_mapping.begin(), label_mapping.end(), label_mapping.begin());

    const NodeID c_n = label_mapping.empty() ? 0 : static_cast<NodeID>(label_mapping.back());

    // send mapping to other PEs that use cluster IDs from this PE -- i.e., answer in_msg
    std::vector<scalable_vector<NodeID>> out_msg(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        out_msg[pe].resize(in_msg[pe].size());
        tbb::parallel_for<std::size_t>(0, in_msg[pe].size(), [&](const std::size_t i) {
            KASSERT(in_msg[pe][i] < label_mapping.size());
            out_msg[pe][i] =
                label_mapping[in_msg[pe][i]] - 1; // label_mapping is 1-based due to the prefix sum operation
        });
    });

    const auto label_remap = mpi::sparse_alltoall_get<NodeID>(std::move(out_msg), graph.communicator());

    // migrate nodes from overloaded PEs
    scalable_vector<GlobalNodeID> c_distribution =
        create_distribution_from_local_count<GlobalNodeID>(c_n, graph.communicator());
    scalable_vector<GlobalNodeID> perfect_distribution{};
    scalable_vector<GlobalNodeID> pe_overload{};
    scalable_vector<GlobalNodeID> pe_underload{};

    if (migrate_nodes) {
        const GlobalNodeID global_c_n = c_distribution.back();
        perfect_distribution          = create_perfect_distribution_from_global_count(global_c_n, graph.communicator());

        // compute diff between perfect distribution and current distribution
        pe_overload.resize(size + 1);
        pe_underload.resize(size + 1);

        scalable_vector<GlobalNodeID> pe_overload_tmp(size);
        scalable_vector<GlobalNodeID> pe_underload_tmp(size);

        for (PEID pe = 0; pe < size; ++pe) {
            const auto [from, to]     = math::compute_local_range<GlobalNodeID>(global_c_n, size, pe);
            const auto balanced_count = static_cast<NodeID>(to - from);
            const auto actual_count   = static_cast<NodeID>(c_distribution[pe + 1] - c_distribution[pe]);

            if (balanced_count > actual_count) {
                pe_underload_tmp[pe] = balanced_count - actual_count;
            } else {
                pe_overload_tmp[pe] = actual_count - balanced_count;
            }
        }

        // prefix sums allow us to find the new owner of a migrating node in log time using binary search
        parallel::prefix_sum(pe_overload_tmp.begin(), pe_overload_tmp.end(), pe_overload.begin() + 1);
        parallel::prefix_sum(pe_underload_tmp.begin(), pe_underload_tmp.end(), pe_underload.begin() + 1);
    }

    // now  we use label_mapping as a [fine node -> coarse node] mapping of local nodes on this PE -- and extend it
    // for ghost nodes in the next step
    // all cluster[.] labels are stored in label_remap, thus we can overwrite label_mapping
    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID u_cluster = clustering[u];
        PEID               u_cluster_owner;
        NodeID             u_local_cluster;

        if (graph.is_owned_global_node(u_cluster)) {
            u_cluster_owner = rank;
            u_local_cluster = graph.global_to_local_node(u_cluster);
        } else {
            u_cluster_owner = graph.find_owner_of_global_node(u_cluster);
            u_local_cluster = static_cast<NodeID>(u_cluster - graph.offset_n(u_cluster_owner));
        }

        tbb::concurrent_hash_map<NodeID, NodeID>::accessor accessor;
        [[maybe_unused]] const bool found = used_clusters_map[u_cluster_owner].find(accessor, u_local_cluster);
        KASSERT(found, V(u_local_cluster) << V(u_cluster_owner) << V(u) << V(u_cluster));

        const NodeID slot_in_msg = accessor->second;
        const NodeID label       = label_remap[u_cluster_owner][slot_in_msg];

        if (migrate_nodes) {
            const auto count =
                static_cast<NodeID>(perfect_distribution[u_cluster_owner + 1] - perfect_distribution[u_cluster_owner]);
            if (label < count) { // node can stay on PE
                label_mapping[u] = perfect_distribution[u_cluster_owner] + label;
            } else { // move node to another PE
                const GlobalNodeID position = pe_overload[u_cluster_owner] + label - count;
                const PEID         new_owner =
                    static_cast<PEID>(math::find_in_distribution<GlobalNodeID>(position, pe_underload));

                KASSERT(position >= pe_underload[new_owner]);
                KASSERT(
                    perfect_distribution[new_owner + 1] - perfect_distribution[new_owner]
                    > c_distribution[new_owner + 1] - c_distribution[new_owner]
                );

                label_mapping[u] = perfect_distribution[new_owner] + c_distribution[new_owner + 1]
                                   - c_distribution[new_owner] + position - pe_underload[new_owner];
            }
        } else {
            label_mapping[u] = c_distribution[u_cluster_owner] + label;
        }
    });

    // exchange labels for ghost nodes
    exchange_ghost_node_mapping(graph, label_mapping);

    if (migrate_nodes) {
        c_distribution = std::move(perfect_distribution);
    }

    return {std::move(label_mapping), std::move(c_distribution)};
}

/*!
 * Construct the coarse graph.
 * @tparam CoarseNodeOwnerCallback
 * @param graph The distributed graph to be contracted.
 * @param mapping Label mapping from fine to coarse nodes.
 * @param c_node_distribution Coarse node distribution: determines which coarse nodes are owned by which PEs using the
 * lambda callback.
 * @param compute_coarse_node_owner Determines which coarse node is owned by which PE: this could be computed using
 * binary search on \c c_node_distribution, but based on the coarse node distribution, the PE could also be computed
 * in constant time.
 * @return The distributed coarse graph.
 */
template <typename CoarseNodeOwnerCallback, typename Mapping>
DistributedGraph build_coarse_graph(
    const DistributedGraph& graph, const Mapping& mapping, scalable_vector<GlobalNodeID> c_node_distribution,
    CoarseNodeOwnerCallback&& compute_coarse_node_owner
) {
    SCOPED_TIMER("Build coarse graph", TIMER_DETAIL);

    const PEID size = mpi::get_comm_size(graph.communicator());
    const PEID rank = mpi::get_comm_rank(graph.communicator());

    // compute coarse node distribution
    const auto from = c_node_distribution[rank];
    const auto to   = c_node_distribution[rank + 1];

    // create messages
    std::vector<NoinitVector<LocalToGlobalEdge>> out_msg(size); // declare outside scope
    {
        SCOPED_TIMER("Create edge messages", TIMER_DETAIL);
        const PEID                                num_threads = omp_get_max_threads();
        std::vector<cache_aligned_vector<EdgeID>> num_messages(num_threads, cache_aligned_vector<EdgeID>(size));

        START_TIMER("Count messages", TIMER_DETAIL);
#pragma omp parallel for default(none) \
    shared(num_messages, graph, mapping, compute_coarse_node_owner, c_node_distribution)
        for (NodeID u = 0; u < graph.n(); ++u) {
            const PEID thread    = omp_get_thread_num();
            const auto c_u       = mapping[u];
            const auto c_u_owner = compute_coarse_node_owner(c_u, c_node_distribution);

            // for (EdgeID e = graph.first_edge(u); e < graph.first_invalid_edge(u); ++e) {
            for (const auto [e, v]: graph.neighbors(u)) {
                // const auto v = graph.edge_target(e);
                const auto c_v        = mapping[v];
                const bool is_message = c_u != c_v;
                num_messages[thread][c_u_owner] += is_message;
            }
        }

        mpi::graph::internal::inclusive_col_prefix_sum(num_messages); // TODO move this utility function somewhere else
        STOP_TIMER(TIMER_DETAIL);

        // allocate send buffers
        START_TIMER("Allocation", TIMER_DETAIL);
        tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msg[pe].resize(num_messages.back()[pe]); });
        STOP_TIMER(TIMER_DETAIL);

        START_TIMER("Create messages", TIMER_DETAIL);
#pragma omp parallel for default(none) \
    shared(num_messages, graph, mapping, compute_coarse_node_owner, c_node_distribution, out_msg)
        for (NodeID u = 0; u < graph.n(); ++u) {
            const PEID thread    = omp_get_thread_num();
            const auto c_u       = mapping[u];
            const auto c_u_owner = compute_coarse_node_owner(c_u, c_node_distribution);
            const auto local_c_u = static_cast<NodeID>(c_u - c_node_distribution[c_u_owner]);

            for (const auto [e, v]: graph.neighbors(u)) {
                const auto c_v = mapping[v];

                if (c_u != c_v) { // ignore self loops
                    const std::size_t slot   = --num_messages[thread][c_u_owner];
                    out_msg[c_u_owner][slot] = {.u = local_c_u, .weight = graph.edge_weight(e), .v = c_v};
                }
            }
        }
        STOP_TIMER(TIMER_DETAIL);
    }

    // deduplicate edges
    TIMED_SCOPE("Deduplicate edges before sending", TIMER_DETAIL) {
        DeduplicateEdgeListMemoryContext deduplicate_m_ctx;
        for (PEID pe = 0; pe < size; ++pe) {
            auto result       = deduplicate_edge_list_parallel(std::move(out_msg[pe]), std::move(deduplicate_m_ctx));
            out_msg[pe]       = std::move(result.first);
            deduplicate_m_ctx = std::move(result.second);
        }
    };

    // exchange messages
    START_TIMER("Exchange edges", TIMER_DETAIL);
    auto in_msg = mpi::sparse_alltoall_get<LocalToGlobalEdge>(std::move(out_msg), graph.communicator());
    STOP_TIMER(TIMER_DETAIL);

    // Copy edge lists to a single list and free old list
    START_TIMER("Copy edge list", TIMER_DETAIL);
    std::vector<std::size_t> in_msg_sizes(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { in_msg_sizes[pe] = in_msg[pe].size(); });
    parallel::prefix_sum(in_msg_sizes.begin(), in_msg_sizes.end(), in_msg_sizes.begin());

    START_TIMER("Allocation", TIMER_DETAIL);
    NoinitVector<LocalToGlobalEdge> edge_list(in_msg_sizes.back());
    STOP_TIMER(TIMER_DETAIL);

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        tbb::parallel_for<std::size_t>(0, in_msg[pe].size(), [&](const std::size_t i) {
            edge_list[in_msg_sizes[pe] - in_msg[pe].size() + i] = in_msg[pe][i];
        });
        // std::copy(in_msg[pe].begin(), in_msg[pe].end(), edge_list.begin() + in_msg_sizes[pe] - in_msg[pe].size());
    });
    STOP_TIMER(TIMER_DETAIL);

    // TODO since we do not know the number of coarse ghost nodes yet, allocate memory only for local nodes and
    // TODO resize in build_distributed_graph_from_edge_list
    KASSERT(from <= to);
    scalable_vector<parallel::Atomic<NodeWeight>> node_weights(to - from);
    struct NodeWeightMessage {
        NodeID     node;
        NodeWeight weight;
    };

    START_TIMER("Exchange node weights", TIMER_DETAIL);
    mpi::graph::sparse_alltoall_custom<NodeWeightMessage>(
        graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER,
        [&](const NodeID u) { return compute_coarse_node_owner(mapping[u], c_node_distribution); },
        [&](const NodeID u) -> NodeWeightMessage {
            const auto   c_u       = mapping[u];
            const PEID   c_u_owner = compute_coarse_node_owner(c_u, c_node_distribution);
            const NodeID c_u_local = c_u - c_node_distribution[c_u_owner];
            return {c_u_local, graph.node_weight(u)};
        },
        [&](const auto r) {
            tbb::parallel_for<std::size_t>(0, r.size(), [&](const std::size_t i) {
                node_weights[r[i].node].fetch_add(r[i].weight, std::memory_order_relaxed);
            });
        }
    );
    STOP_TIMER(TIMER_DETAIL);

    // now every PE has an edge list with all edges -- so we can build the graph from it
    return build_distributed_graph_from_edge_list(
        edge_list, std::move(c_node_distribution), graph.communicator(),
        [&](const NodeID u) {
            KASSERT(u < node_weights.size());
            return node_weights[u].load(std::memory_order_relaxed);
        },
        compute_coarse_node_owner
    );
}

/*!
 * Construct the coarse graph.
 * @param graph The distributed graph to be contracted.
 * @param mapping Label mapping from fine to coarse nodes.
 * @param c_node_distribution Coarse node distribution: determines which coarse nodes are owned by which PEs using
 * binary search.
 * @return The distributed coarse graph.
 */
template <typename Mapping>
DistributedGraph build_coarse_graph(
    const DistributedGraph& graph, const Mapping& mapping, scalable_vector<GlobalNodeID> c_node_distribution
) {
    return build_coarse_graph(
        graph, mapping, c_node_distribution,
        [](const GlobalNodeID node, const auto& node_distribution) {
            const auto it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), node);
            return static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
        }
    );
}

/*!
 * Sparse all-to-all to update ghost node weights after coarse graph construction.
 * @param graph Distributed graph with invalid ghost node weights.
 */
void update_ghost_node_weights(DistributedGraph& graph) {
    SCOPED_TIMER("Update ghost node weights", TIMER_DETAIL);

    struct Message {
        NodeID     local_node;
        NodeWeight weight;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<Message>(
        graph,
        [&](const NodeID u) -> Message {
            return {u, graph.node_weight(u)};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_other_pe, weight] = buffer[i];
                const NodeID local_node = graph.global_to_local_node(graph.offset_n(pe) + local_node_on_other_pe);
                graph.set_ghost_node_weight(local_node, weight);
            });
        }
    );
}
} // namespace

//! Contract a distributed graph such that coarse nodes are owned by the PE which owned the respective cluster ID.
GlobalContractionResult
contract_global_clustering_no_migration(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SCOPED_TIMER("Contract clustering");

    auto [mapping, distribution] = compute_mapping(graph, clustering);
    auto c_graph                 = build_coarse_graph(graph, mapping, std::move(distribution));
    update_ghost_node_weights(c_graph);

    return {std::move(c_graph), std::move(mapping)};
}

//! Contract a distributed graph such that *most* coarse nodes are owned by the PE which owned the respective cluster
//! ID, while migrating enough coarse nodes such that each PE ownes approx. the same number of coarse nodes.
GlobalContractionResult
contract_global_clustering_minimal_migration(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SCOPED_TIMER("Contract clustering");

    auto [mapping, distribution] = compute_mapping(graph, clustering, true);
    auto c_graph                 = build_coarse_graph(graph, mapping, std::move(distribution));
    update_ghost_node_weights(c_graph);

    return {std::move(c_graph), std::move(mapping)};
}

//! Contract a distributed graph such that each PE owns the same number of coarse nodes by assigning coarse nodes
//! \code{p*n/s .. (p + 1)*n/s} to PE \c p, where \c n is the number of coarse nodes and \c s is the number of PEs.
GlobalContractionResult
contract_global_clustering_full_migration(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SCOPED_TIMER("Contract clustering");

    auto [mapping, distribution] = compute_mapping(graph, clustering);

    // create a new node distribution where nodes are evenly distributed across PEs
    const PEID         size       = mpi::get_comm_size(graph.communicator());
    const GlobalNodeID c_global_n = distribution.back();
    auto               c_graph    = build_coarse_graph(
                         graph, mapping, create_perfect_distribution_from_global_count<GlobalNodeID>(c_global_n, graph.communicator()),
                         [size, c_global_n](const GlobalNodeID node, const auto& /* node_distribution */) {
            return math::compute_local_range_rank<GlobalNodeID>(c_global_n, size, node);
        }
                     );

    update_ghost_node_weights(c_graph);

    return {std::move(c_graph), std::move(mapping)};
}

GlobalContractionResult contract_global_clustering(
    const DistributedGraph& graph, const GlobalClustering& clustering, const GlobalContractionAlgorithm algorithm
) {
    switch (algorithm) {
        case GlobalContractionAlgorithm::NO_MIGRATION:
            return contract_global_clustering_no_migration(graph, clustering);
        case GlobalContractionAlgorithm::MINIMAL_MIGRATION:
            return contract_global_clustering_minimal_migration(graph, clustering);
        case GlobalContractionAlgorithm::FULL_MIGRATION:
            return contract_global_clustering_full_migration(graph, clustering);
    }
    __builtin_unreachable();
}

/*!
 * Projects the partition of the coarse graph onto the fine graph. Works for any graph contraction variations.
 * @param fine_graph The distributed fine graph.
 * @param coarse_graph The distributed coarse graph with partition.
 * @param fine_to_coarse Mapping from fine to coarse nodes.
 * @return Projected partition of the fine graph.
 */
DistributedPartitionedGraph project_global_contracted_graph(
    const DistributedGraph& fine_graph, DistributedPartitionedGraph coarse_graph, const GlobalMapping& fine_to_coarse
) {
    SCOPED_TIMER("Project partition");

    const PEID size = mpi::get_comm_size(fine_graph.communicator());

    // find unique coarse_graph node IDs of fine_graph nodes
    auto resolve_coarse_node = [&](const GlobalNodeID coarse_node) {
        KASSERT(coarse_node < coarse_graph.global_n());
        const PEID owner = coarse_graph.find_owner_of_global_node(coarse_node);
        const auto local = static_cast<NodeID>(coarse_node - coarse_graph.offset_n(owner));
        return std::make_pair(owner, local);
    };

    auto used_coarse_nodes = find_used_cluster_ids_per_pe(fine_graph, fine_to_coarse, resolve_coarse_node);

    auto& used_coarse_nodes_map = used_coarse_nodes.first;
    auto& used_coarse_nodes_vec = used_coarse_nodes.second;

    // send requests for block IDs
    const auto reqs = mpi::sparse_alltoall_get<NodeID>(used_coarse_nodes_vec, fine_graph.communicator());

    // build response messages
    START_TIMER("Allocation", TIMER_DETAIL);
    std::vector<scalable_vector<BlockID>> resps(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { resps[pe].resize(reqs[pe].size()); });
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Build response messages", TIMER_DETAIL);
    tbb::parallel_for<std::size_t>(0, reqs.size(), [&](const std::size_t i) {
        tbb::parallel_for<std::size_t>(0, reqs[i].size(), [&](const std::size_t j) {
            KASSERT(coarse_graph.is_owned_node(reqs[i][j]));
            resps[i][j] = coarse_graph.block(reqs[i][j]);
        });
    });
    STOP_TIMER(TIMER_DETAIL);

    // exchange messages and use used_coarse_nodes_map to store block IDs
    static_assert(std::numeric_limits<BlockID>::digits <= std::numeric_limits<NodeID>::digits);
    mpi::sparse_alltoall<BlockID>(
        std::move(resps),
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                KASSERT(static_cast<std::size_t>(pe) < used_coarse_nodes_map.size());
                KASSERT(static_cast<std::size_t>(pe) < reqs.size());
                KASSERT(i < used_coarse_nodes_vec[pe].size());

                UsedClustersMap::accessor   accessor;
                [[maybe_unused]] const bool found =
                    used_coarse_nodes_map[pe].find(accessor, used_coarse_nodes_vec[pe][i]);
                KASSERT(found);
                accessor->second = buffer[i];
            });
        },
        fine_graph.communicator()
    );

    // assign block IDs to fine nodes
    START_TIMER("Allocation", TIMER_DETAIL);
    scalable_vector<parallel::Atomic<BlockID>> fine_partition(fine_graph.total_n());
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Set blocks", TIMER_DETAIL);
    fine_graph.pfor_nodes([&](const NodeID u) {
        const auto [owner, local] = resolve_coarse_node(fine_to_coarse[u]);

        UsedClustersMap::accessor   accessor;
        [[maybe_unused]] const bool found = used_coarse_nodes_map[owner].find(accessor, local);
        KASSERT(found);

        fine_partition[u] = accessor->second;
    });
    STOP_TIMER(TIMER_DETAIL);

    // exchange ghost node labels
    struct GhostNodeLabel {
        NodeID  local_node_on_sender;
        BlockID block;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<GhostNodeLabel>(
        fine_graph,
        [&](const NodeID u) -> GhostNodeLabel {
            return {u, fine_partition[u]};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_sender, block] = buffer[i];
                const GlobalNodeID global_node            = fine_graph.offset_n(pe) + local_node_on_sender;
                const NodeID       local_node             = fine_graph.global_to_local_node(global_node);
                fine_partition[local_node]                = block;
            });
        }
    );

    return {&fine_graph, coarse_graph.k(), std::move(fine_partition), coarse_graph.take_block_weights()};
}
} // namespace kaminpar::dist
