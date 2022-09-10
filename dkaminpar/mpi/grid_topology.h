/*******************************************************************************
 * @file:   grid_topology.h
 * @author: Daniel Seemaier
 * @date:   09.09.2022
 * @brief:
 ******************************************************************************/
#pragma once

#include <cmath>

#include "dkaminpar/definitions.h"

#include "common/logger.h"

namespace kaminpar::mpi {
class GridTopology {
    SET_DEBUG(true);

public:
    GridTopology(const PEID size) : _size(size), _sqrt(static_cast<PEID>(std::sqrt(size))) {}

    inline PEID row(const PEID pe) const {
        if (pe < num_pes_in_full_rectangle()) {
            return pe / num_cols();
        } else {
            return (pe - num_pes_in_full_rectangle()) / num_full_columns() + partial_column_size();
        }
    }

    inline PEID col(const PEID pe) const {
        if (pe < num_pes_in_full_rectangle()) {
            return pe % num_cols();
        } else {
            return (pe - num_pes_in_full_rectangle()) % num_full_columns();
        }
    }

    inline PEID virtual_col(const PEID pe) const {
        if (partial_column_size() == 0 || col(pe) < num_full_columns()) {
            return col(pe);
        } else {
            return row(pe);
        }
    }

    inline PEID row_size(const PEID row) const {
        if (row < partial_column_size()) {
            return num_cols();
        } else {
            return num_full_columns();
        }
    }

    inline PEID col_size(const PEID column) const {
        if (column < num_full_columns()) {
            return _sqrt;
        } else {
            return partial_column_size();
        }
    }

    inline PEID virtual_col_size(const PEID column) const {
        return _sqrt + (column < partial_column_size());
    }

    inline PEID num_cols() const {
        return std::ceil(1.0 * _size / _sqrt);
    }

    inline PEID num_rows() const {
        return _sqrt;
    }

    inline PEID num_cols_in_row(const PEID row) const {
        return num_full_columns() + (row < partial_column_size());
    }

private:
    inline PEID num_full_columns() const {
        return _size / _sqrt;
    }

    inline PEID partial_column_size() const {
        return _size - _sqrt * num_full_columns();
    }

    inline PEID num_pes_in_full_rectangle() const {
        return partial_column_size() == 0 ? _size : num_cols() * partial_column_size();
    }

    PEID _size;
    PEID _sqrt;
};
} // namespace kaminpar::mpi