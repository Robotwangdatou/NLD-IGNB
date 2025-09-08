# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 12:35:52 2025
The code implementation of NLD-IGNB
@author: robot
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize, StandardScaler


# -------------------------- 1. Basic Utility Functions --------------------------
# Calculate G-mean (a key metric for evaluating imbalanced data)
def g_mean_score(y_true, y_pred):   
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    sensitivities = np.nan_to_num(sensitivities, nan=0.0)  # Avoid NaN from empty classes
    g_mean = np.sqrt(np.prod(sensitivities))
    return g_mean

# Calculate evaluation metrics (supports binary/multi-class classification, focusing on minority classes)
def calculate_metrics(y, y_predict, y_prob, minority_class):
    """Returns 3 core metrics: AUC, Gmean, F1"""
    if len(np.unique(y)) == 2:
        # Binary classification: focus on minority class (positive class)
        pos_label = minority_class
        auc = roc_auc_score(y, y_prob[:, np.where(np.unique(y)==pos_label)[0][0]])
    else:
        # Multi-class classification: use macro averaging
        y_bin = label_binarize(y, classes=np.unique(y))
        auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
    
    g_mean = g_mean_score(y_true=y, y_pred=y_pred)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    return [auc, g_mean, f1]

# Calculate weighted neighbor statistics (weight-optimized local parameters)
def calculate_weighted_local_stats(knn_model, point, k, X_class, weight_by_distance=True):
    """
    Calculate weighted local median (location) and weighted variance (scale) for a single sample
    Closer samples get higher weights to reduce noise and distant neighbor interference
    """
    if knn_model is None or X_class is None or len(X_class) == 0:
        return None
    
    # Get distances and indices of k nearest neighbors
    distances, indices = knn_model.kneighbors(point.reshape(1, -1), n_neighbors=k)
    indices = indices[0]
    distances = distances[0]
    
    # Extract neighbor samples
    neighbor_points = X_class.iloc[indices] if isinstance(X_class, pd.DataFrame) else X_class[indices]
    
    # Calculate weights (inverse distance, avoid division by zero)
    if weight_by_distance and np.sum(distances) > 0:
        weights = 1 / (distances + 1e-8)
        weights = weights / np.sum(weights)  # Normalization
    else:
        weights = np.ones(k) / k  # Equal weights
    
    # Calculate weighted median (resistant to outliers)
    weighted_median = np.zeros(neighbor_points.shape[1])
    for col_idx in range(neighbor_points.shape[1]):
        sorted_vals = np.sort(neighbor_points[:, col_idx] if isinstance(neighbor_points, np.ndarray) else neighbor_points.iloc[:, col_idx])
        sorted_weights = weights[np.argsort(neighbor_points[:, col_idx] if isinstance(neighbor_points, np.ndarray) else neighbor_points.iloc[:, col_idx])]
        cum_weights = np.cumsum(sorted_weights)
        weighted_median[col_idx] = sorted_vals[np.argmax(cum_weights >= 0.5)]  # Find position where cumulative weight exceeds 0.5
    
    # Calculate weighted variance (more robust scale estimation)
    weighted_mean = np.average(neighbor_points, axis=0, weights=weights)
    weighted_var = np.average((neighbor_points - weighted_mean) **2, axis=0, weights=weights)
    
    return {'median': weighted_median, 'var': weighted_var}

# Calculate neighbor class distribution (to correct prior probabilities)
def calculate_neighbor_class_dist(knn_all, x, k, train_y):
    """Calculate class distribution of sample neighbors for dynamically correcting prior probabilities"""
    distances, neighbor_indices = knn_all.kneighbors(x.reshape(1, -1), n_neighbors=k)
    neighbor_classes = train_y.iloc[neighbor_indices[0]] if isinstance(train_y, pd.Series) else train_y[neighbor_indices[0]]
    class_counts = np.bincount(neighbor_classes, minlength=len(np.unique(train_y)))
    return class_counts / np.sum(class_counts)  # Local class proportion


# -------------------------- 2. Two-direction Fusion Improved GNB: NLD-IGNB --------------------------
class IntegratedGNB:
    """
    GNB integrated with two major improvements:
    1. Neighbor class distribution corrected prior (direction 3): dynamically adjust priors to adapt to imbalanced data
    2. Weighted neighbor statistics (direction 2): improve stability of local parameter (median+variance) estimation
    """
    def __init__(self, 
                 # Neighbor parameters (shared)
                 k_local_factor=0.2,    # Neighbor proportion for local parameter estimation (k_local = n_samples × this value)
                 k_prior_factor=0.1,   # Neighbor proportion for prior correction (k_prior = n_samples × this value)
                 min_k=5, max_k=30,     # Neighbor count bounds
                 # Weight parameters
                 weight_strength=1.0,   # Distance weight strength (1.0 is default, >1 enhances weight differences)
                 # Prior correction parameters
                 prior_weight=0.7,      # Local distribution weight (global weight = 1-prior_weight)
                 # Threshold parameters
                 minority_threshold=0.5, # Minority class classification threshold (default 0.3, lower than 0.5 is more lenient)
                 # Basic parameters
                 var_smoothing=1e-9):
        # Neighbor parameters
        self.k_local_factor = k_local_factor
        self.k_prior_factor = k_prior_factor
        self.min_k = min_k
        self.max_k = max_k
        # Weight parameters
        self.weight_strength = weight_strength
        # Prior correction parameters
        self.prior_weight = prior_weight
        # Thresholds
        self.minority_threshold = minority_threshold
        self.minority_class = None  # Store minority class label (determined during training)
        # Basic parameters
        self.var_smoothing = var_smoothing
        
        # Model parameters (assigned after training)
        self.classes_ = None
        self.global_prior_ = None          # Global prior
        self.global_median_ = None         # Global median (fallback)
        self.global_var_ = None            # Global variance (fallback)
        self.class_knn_models_ = {}        # KNN for each class (for local parameter estimation)
        self.knn_all_ = None               # KNN for entire training set (for prior correction)
        self.class_samples_ = {}           # Samples for each class
        self.train_y_ = None               # Training set labels
        self.k_local_opt_ = None           # Optimized neighbor count for local parameters
        self.k_prior_opt_ = None           # Optimized neighbor count for prior correction

    def fit(self, X, y):
        """Training function: initialize all model parameters"""
        X = np.asarray(X)
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        self.train_y_ = y
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Identify minority class
        self.minority_class = y.value_counts().idxmin()
        
        # 1. Calculate global prior (basis for direction 3)
        class_counts = np.bincount(y, minlength=n_classes)
        self.global_prior_ = class_counts / np.sum(class_counts)
        
        # 2. Initialize KNN models for each class and global statistics (basis for direction 2)
        self.global_median_ = np.zeros((n_classes, n_features))
        self.global_var_ = np.zeros((n_classes, n_features))
        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.class_samples_[cls] = X_c
            # Global median and variance (fallback)
            self.global_median_[i] = np.median(X_c, axis=0)
            self.global_var_[i] = np.var(X_c, axis=0) + self.var_smoothing
            # Build in-class KNN model (for local parameter estimation)
            if len(X_c) >= self.min_k:
                self.class_knn_models_[cls] = NearestNeighbors(algorithm='ball_tree').fit(X_c)
            else:
                self.class_knn_models_[cls] = None
        
        # 3. Build KNN for entire training set (for prior correction, direction 3)
        self.knn_all_ = NearestNeighbors(algorithm='ball_tree').fit(X)
        
        # 4. Determine optimized neighbor counts
        self.k_local_opt_ = max(self.min_k, min(int(n_samples * self.k_local_factor), self.max_k))
        self.k_prior_opt_ = max(self.min_k, min(int(n_samples * self.k_prior_factor), self.max_k))
        print(f"✅ Training completed | Local parameter k={self.k_local_opt_} | Prior correction k={self.k_prior_opt_} ")
        
        return self

    def _get_local_params(self, x, c):
        """Get weighted local parameters (weighted neighbor statistics)"""
        X_c = self.class_samples_.get(c, None)
        knn_model = self.class_knn_models_.get(c, None)
        if X_c is None or knn_model is None or len(X_c) < self.min_k:
            # Use global parameters when insufficient samples
            c_idx = np.where(self.classes_ == c)[0][0]
            return self.global_median_[c_idx], self.global_var_[c_idx]
        
        # Calculate weighted local statistics
        local_stats = calculate_weighted_local_stats(
            knn_model, x, self.k_local_opt_, X_c, weight_by_distance=True
        )
        mu = local_stats['median']
        sigma = local_stats['var'] + self.var_smoothing  # Smoothing
        return mu, sigma

    def _get_corrected_prior(self, x):
        """Correct prior probabilities (fuse global and local)"""
        # Calculate local class distribution
        local_dist = calculate_neighbor_class_dist(
            self.knn_all_, x, self.k_prior_opt_, self.train_y_
        )
        # Weighted fusion (global×(1-w) + local×w)
        corrected_prior = (self.global_prior_ * (1 - self.prior_weight)) + (local_dist * self.prior_weight)
        return corrected_prior / np.sum(corrected_prior)  # Normalization

    def predict_proba(self, X):
        """Predict posterior probabilities: fuse local parameters and corrected priors"""
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            x = X[i]
            # 1. Get corrected prior (direction 3)
            corrected_prior = self._get_corrected_prior(x)
            
            # 2. Calculate conditional probabilities with weighted local parameters (direction 2)
            log_cond_prob = np.zeros(n_classes)
            for j, cls in enumerate(self.classes_):
                mu, sigma = self._get_local_params(x, cls)
                log_p = -0.5 * (np.log(2 * np.pi) + np.log(sigma)) - 0.5 * ((x - mu)** 2) / sigma
                log_cond_prob[j] = np.sum(log_p)
            
            # 3. Calculate posterior probabilities
            log_posterior = log_cond_prob + np.log(corrected_prior + 1e-12)  # Avoid log(0)
            log_posterior -= np.max(log_posterior)  # Numerical stability
            proba[i] = np.exp(log_posterior) / np.sum(np.exp(log_posterior))
        
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            
            pos_idx = np.where(self.classes_ == self.minority_class)[0][0]
            neg_idx = 1 - pos_idx
            # Threshold judgment
            predictions = np.where(
                proba[:, pos_idx] >= self.minority_threshold,
                self.minority_class,
                self.classes_[neg_idx]
            )
            return predictions
       


# -------------------------- 3. Main Process: Cross-validation Evaluation --------------------------
if __name__ == '__main__':
    # 1. Data loading (example: imbalanced dataset)
    try:
        dataset = pd.read_csv('glass1.csv')  # Replace with your dataset
        print(f"📊 Data loaded successfully | Shape: {dataset.shape}")
        print(f"Class distribution:\n{dataset['class'].value_counts()}")
        
        # Missing value handling (consistent with local median imputation strategy)
        if dataset.isnull().any().any():
            print("⚠️ Missing values detected, filling with feature medians")
            for col in dataset.columns:
                if col != 'class' and dataset[col].isnull().any():
                    dataset[col].fillna(dataset[col].median(), inplace=True)
    except FileNotFoundError:
        print("❌ Error: Data file not found, please check the path!")
        exit()

    # 2. Data splitting
    y = dataset['class']
    X_original = dataset.drop(columns='class')
    minority_class = y.value_counts().idxmin()
    print(f"🔍 Minority class label: {minority_class} | Proportion: {np.mean(y == minority_class):.2%}")

    # 3. Cross-validation configuration
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = pd.DataFrame(columns=['Fold', 'AUC', 'Gmean', 'F1'])

    # 4. Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_original, y), 1):
        print(f"\n=== Fold {fold}/{n_splits} training ===")
        
        # Split into training/test sets
        X_train, X_test = X_original.iloc[train_idx], X_original.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardization (GNB is sensitive to scale, must standardize)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize integrated improved GNB
        integrated_gnb = IntegratedGNB(
            # Neighbor parameters
            k_local_factor=0.2,    # Proportion for local parameter estimation (tunable)
            k_prior_factor=0.1,   # Proportion for prior correction (tunable)
            min_k=5, max_k=30,
            # Weight parameters
            weight_strength=1.0,   # Distance weight strength (>1 enhances neighbor influence)
            # Prior correction parameters
            prior_weight=0.7,      # Local prior weight (0.5~0.9)
            # Threshold
            minority_threshold=0.5,
            var_smoothing=1e-9
        )
        integrated_gnb.fit(X_train_scaled, y_train)

        # Prediction and evaluation
        y_pred = integrated_gnb.predict(X_test_scaled)
        y_prob = integrated_gnb.predict_proba(X_test_scaled)
        metrics = calculate_metrics(y_test, y_pred, y_prob, minority_class)

        # Record results
        fold_result = pd.DataFrame({
            'Fold': [fold],
            'AUC': [metrics[0]],
            'Gmean': [metrics[1]],
            'F1': [metrics[2]]
        })
        results = pd.concat([results, fold_result], ignore_index=True)

        # Print current fold metrics
        print(f"  Fold {fold} metrics:")
        print(f"    AUC: {metrics[0]:.4f} | Gmean: {metrics[1]:.4f} | F1: {metrics[2]:.4f}")

    # 5. Output final results
    print("\n" + "="*100)
    print("📋 5-fold cross-validation results for NLD-IGNB")
    print("="*100)
    print("\n[Per-fold detailed metrics]")
    print(results.round(4))
    
    avg_metrics = results[['AUC', 'Gmean', 'F1']].mean()
    std_metrics = results[['AUC', 'Gmean', 'F1']].std()
    print("\n[5-fold average metrics]")
    
    for metric in avg_metrics.index:
        print(f"{metric:10s}: {avg_metrics[metric]:.4f} (±{std_metrics[metric]:.4f})")
    print("="*80)

