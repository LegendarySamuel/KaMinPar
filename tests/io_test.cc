#include <gmock/gmock.h>

#include "tests.h"

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/io.h"

using ::testing::Eq;
using ::testing::UnorderedElementsAre;
using namespace kaminpar;

inline void assert_K3_structure(const Graph& G) {
    ASSERT_THAT(G.n(), Eq(3));
    ASSERT_THAT(G.m(), Eq(6));
    ASSERT_THAT(test::view_to_vector(G.adjacent_nodes(0)), UnorderedElementsAre(1, 2));
    ASSERT_THAT(test::view_to_vector(G.adjacent_nodes(1)), UnorderedElementsAre(0, 2));
    ASSERT_THAT(test::view_to_vector(G.adjacent_nodes(2)), UnorderedElementsAre(0, 1));
}

inline auto outgoing_edge_weights(const Graph& G, const NodeID u) {
    std::vector<EdgeWeight> edge_weights;
    for (const EdgeID& e: G.incident_edges(u)) {
        edge_weights.push_back(G.edge_weight(e));
    }
    return edge_weights;
}

TEST(IOTest, unweighted_K3) {
    const auto G = io::metis::read(test::test_instance("unweighted_K3.graph"));
    assert_K3_structure(G);

    for (const NodeID& u: G.nodes())
        ASSERT_THAT(G.node_weight(u), Eq(1));
    for (const EdgeID& e: G.edges())
        ASSERT_THAT(G.edge_weight(e), Eq(1));
}

TEST(IOTest, node_weighted_K3) {
    const auto G = io::metis::read(test::test_instance("node_weighted_K3.graph"));
    assert_K3_structure(G);

    ASSERT_THAT(G.node_weight(0), Eq(1));
    ASSERT_THAT(G.node_weight(1), Eq(2));
    ASSERT_THAT(G.node_weight(2), Eq(3));
    for (const EdgeID& e: G.edges())
        ASSERT_THAT(G.edge_weight(e), Eq(1));
}

TEST(IOTest, edge_weighted_K3) {
    const auto G = io::metis::read(test::test_instance("edge_weighted_K3.graph"));
    assert_K3_structure(G);

    ASSERT_THAT(outgoing_edge_weights(G, 0), UnorderedElementsAre(1, 2));
    ASSERT_THAT(outgoing_edge_weights(G, 1), UnorderedElementsAre(1, 3));
    ASSERT_THAT(outgoing_edge_weights(G, 2), UnorderedElementsAre(2, 3));
}

TEST(IOTest, weighted_K3) {
    const auto G = io::metis::read(test::test_instance("weighted_K3.graph"));
    assert_K3_structure(G);

    ASSERT_THAT(G.node_weight(0), Eq(10));
    ASSERT_THAT(G.node_weight(1), Eq(20));
    ASSERT_THAT(G.node_weight(2), Eq(30));

    ASSERT_THAT(outgoing_edge_weights(G, 0), UnorderedElementsAre(1, 2));
    ASSERT_THAT(outgoing_edge_weights(G, 1), UnorderedElementsAre(1, 3));
    ASSERT_THAT(outgoing_edge_weights(G, 2), UnorderedElementsAre(2, 3));
}

TEST(IOTest, large_weights) {
    const auto G = io::metis::read(test::test_instance("large_weights.graph"));

    ASSERT_THAT(G.node_weight(0), Eq(123456789));
    ASSERT_THAT(G.node_weight(1), Eq(234567891));
}

TEST(IOTest, graph_with_comments) {
    const auto G = io::metis::read(test::test_instance("with_comments.graph"));

    ASSERT_THAT(G.n(), Eq(2));
    ASSERT_THAT(G.m(), Eq(2));
    ASSERT_THAT(test::view_to_vector(G.adjacent_nodes(0)), UnorderedElementsAre(1));
    ASSERT_THAT(test::view_to_vector(G.adjacent_nodes(1)), UnorderedElementsAre(0));
}
