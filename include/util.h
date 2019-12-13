#ifndef _UTIL_H
#define _UTIL_H

#define ARMA_USE_SUPERLU

#include <armadillo>
#include <iostream>

arma::sp_mat spdiags(const arma::mat& B, const std::vector<int>& d, int m, int n)
{
    arma::sp_mat A(m, n);
    for (int k = 0; k < d.size(); k++) {
        int i_min = std::max(0, -d[k]);
        int i_max = std::min(m - 1, n - d[k] - 1);
        A.diag(d[k]) = B(arma::span(0, i_max - i_min), arma::span(k, k));
    }

    return A;
}

#endif