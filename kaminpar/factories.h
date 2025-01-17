/*******************************************************************************
 * @file:   factories.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Factory functions to instantiate coarsening and local improvement
 * algorithms.
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/coarsener.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm::factory {
std::unique_ptr<Coarsener> create_coarsener(const Graph &graph, const CoarseningContext &c_ctx);

std::unique_ptr<Refiner> create_refiner(const Context &ctx);
} // namespace kaminpar::shm::factory
