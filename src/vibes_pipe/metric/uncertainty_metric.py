import numpy as np 


def calculate_uncertainty_metrics(samples, mean_pred, std_pred: None):
    """
    Calculate comprehensive uncertainty metrics including GED
    
    Args:
        samples: MC samples [n_samples, batch, channels, H, W, D]
        mean_pred: Mean prediction
        std_pred: Standard deviation
        
    Returns:
        Dictionary of uncertainty metrics
    """
    eps = 1e-8
    
    # Get the first item from batch (since batch_size=1)
    samples_squeezed = samples[:, 0, 0]  # [n_samples, H, W, D]
    mean_squeezed = mean_pred[0, 0]
    std_squeezed = std_pred[0, 0]
    
    # 1. Predictive Entropy (Total Uncertainty)
    p = mean_squeezed
    predictive_entropy = -(p * np.log(p + eps) + (1-p) * np.log(1-p + eps))
    
    # 2. Mutual Information (Epistemic Uncertainty)
    # MI = H[E[p]] - E[H[p]]
    entropy_of_mean = -(p * np.log(p + eps) + (1-p) * np.log(1-p + eps))
    
    mean_entropy = 0
    for sample in samples_squeezed:
        sample_entropy = -(sample * np.log(sample + eps) + (1-sample) * np.log(1-sample + eps))
        mean_entropy += sample_entropy
    mean_entropy /= len(samples_squeezed)
    
    mutual_info = entropy_of_mean - mean_entropy
    
    # 3. Variance
    variance = np.var(samples_squeezed, axis=0)
    
    # 4. Generalized Energy Distance (GED)
    ged = calculate_ged(samples_squeezed)
    
    # 5. Coefficient of Variation
    cv = std_squeezed / (mean_squeezed + eps)
    
    # 6. Sample Diversity (using pairwise distances)
    diversity = calculate_sample_diversity(samples_squeezed)
    
    return {
        'predictive_entropy': predictive_entropy,
        'mutual_information': mutual_info,
        'variance': variance,
        'std': std_squeezed,
        'ged': ged,
        'coefficient_variation': cv,
        'sample_diversity': diversity
    }


def calculate_ged(samples):
    """
    Calculate Generalized Energy Distance
    
    Args:
        samples: Array of shape [n_samples, H, W, D]
        sa
    Returns:
        GED value
    """
    n_samples = samples.shape[0]
    
    if n_samples < 2:
        return 0.0
    
    # Flatten spatial dimensions for efficiency
    samples_flat = samples.reshape(n_samples, -1)
    
    # Calculate pairwise distances
    ged_sum = 0
    count = 0
    
    # Limit computation for efficiency (use subset if many samples)
    max_pairs = min(n_samples, 15)
    
    for i in range(max_pairs):
        for j in range(i+1, max_pairs):
            # L2 distance
            dist = np.sqrt(np.mean((samples_flat[i] - samples_flat[j]) ** 2))
            ged_sum += dist
            count += 1
    
    ged = ged_sum / max(count, 1)
    
    return ged


def calculate_sample_diversity(samples):
    """
    Calculate diversity among samples using average pairwise correlation.
    This sample diversity computation is basically a Pearson Correlation-based metric.
    
    Args:
        samples: Array of shape [n_samples, H, W, D]
        
    Returns:
        Diversity score (1 - average correlation)
    """
    n_samples = samples.shape[0]
    
    if n_samples < 2:
        return 0.0
    
    # Flatten samples
    samples_flat = samples.reshape(n_samples, -1)
    
    # Calculate pairwise correlations
    correlations = []
    for i in range(min(n_samples, 10)):
        for j in range(i+1, min(n_samples, 10)):
            corr = np.corrcoef(samples_flat[i], samples_flat[j])[0, 1]
            correlations.append(corr)
    
    # Diversity is 1 - average correlation
    avg_correlation = np.mean(correlations)
    diversity = 1 - abs(avg_correlation)
    
    return diversity