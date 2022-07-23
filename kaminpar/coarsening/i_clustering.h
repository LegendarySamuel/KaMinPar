/*******************************************************************************
 * @file:   i_clustering_algorithm.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 * @brief:  Interface for clustering algorithms used for coarsening.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"

#include "common/parallel/atomic.h"

namespace kaminpar::shm {
class IClustering {
public:
    using AtomicClusterArray = scalable_vector<parallel::Atomic<NodeID>>;

    IClustering()          = default;
    virtual ~IClustering() = default;

    IClustering(const IClustering&)                = delete;
    IClustering& operator=(const IClustering&)     = delete;
    IClustering(IClustering&&) noexcept            = default;
    IClustering& operator=(IClustering&&) noexcept = default;

    //
    // Optional options
    //

    virtual void set_max_cluster_weight(const NodeWeight /* weight */) {}
    virtual void set_desired_cluster_count(const NodeID /* count */) {}

    //
    // Clustering function
    //

    virtual const AtomicClusterArray& compute_clustering(const Graph& graph) = 0;
};
} // namespace kaminpar::shm
