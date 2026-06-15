Privacy-utility trade-off metrics
#################################

In order to provide users with metrics to quantify the quality of the noised data generated using *trasgoDP*, particularly in terms of distributional consistency, we have implemented specific functions:

1. First, the most intuitive approach is to compute the divergence between the original column and the one obtained after applying DP, in order to quantify the information loss. To this end, different divergence metrics can be calculated, including Total Variation Distance (TVD), Jensen-Shannon divergence (JS), and Kullback-Leibler divergence (KL).

2. In addition, within *trasgoDP* a novel metric to quantify the correlation loss (expressed as a percentage) is proposed. The idea is to assess how the correlation between a given column and a set of other features changes after applying differential privacy. In this way, large changes in correlations result in higher loss values. This metric is implemented as follows:

.. note::
 
   **Correlation loss (%)**
   
   1. Be :math:`D \in \mathbb{R}^{n \times d}` the original dataset and :math:`D' \in \mathbb{R}^{n \times d+1}` the privatized one (if one new column transformed with DP), or :math:`D' \in \mathbb{R}^{n \times d}` (if the privatized column has been substituted). Let's assume that :math:`D' \in \mathbb{R}^{n \times d}`. Be :math:`F = \{X_1, \dots, X_f\}` the set of features selected, :math:`f \leq n`.
    
   2. For each categorical feature :math:`X_j \in F`, we define a function :math:`\phi_j : \mathcal{C}_j \to \mathbb{Z}` with :math:`\mathcal{C}_j` the set of values observed in :math:`D` and :math:`D'`. Then we get the transformed datasets :math:`\tilde{D}` and :math:`\tilde{D}'`.
    
   3. We extract the matrix based on the selected features: :math:`X = \tilde{D}[F],` :math:`X' = \tilde{D}'[F]`.
    
   4. Be :math:`\rho(\cdot,\cdot)` a correlation method (Pearson, Spearman o Kendall). Then we calculate the correlation matrix as follows: :math:`R = (\rho_{ij})_{i,j=1}^d` with :math:`\rho_{ij} = \rho(X_i, X_j)` and :math:`R' = (\rho'_{ij})_{i,j=1}^d` with :math:`\rho'_{ij} = \rho(X'_i, X'_j)`.

   5. Remove autocorrelation: :math:`\mathcal{I} = \{(i,j)\,:\, i \neq j,\; 1 \leq i,j \leq d\}`.

   6. Difference between correlations: :math:`\Delta_{ij} = |\rho_{ij} - \rho'_{ij}|, \quad (i,j) \in \mathcal{I}`.

   7. Mean of the difference between correlations: :math:`\mu_{\Delta} = \frac{1}{|\mathcal{I}|} \sum_{(i,j)\in \mathcal{I}} |\rho_{ij} - \rho'_{ij}|`, and mean of the original correlation matrix :math:`\mu_{R} = \frac{1}{|\mathcal{I}|} \sum_{(i,j)\in \mathcal{I}} |\rho_{ij}|`.

   8. Correlation loss (\%): :math:`\mathcal{L} = 100 \cdot \frac{\mu_{\Delta}}{\mu_{R}}`.


