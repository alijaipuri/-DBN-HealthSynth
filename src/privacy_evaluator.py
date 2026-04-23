import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, wasserstein_distance
import logging

logger = logging.getLogger(__name__)

class PrivacyEvaluator:
    def __init__(self):
        self.results = {}

    def membership_inference_risk(self, real_data, synthetic_data):
        """Estimate membership inference attack risk (lower = better privacy)"""
        scaler = StandardScaler()
        real_s = scaler.fit_transform(real_data)
        syn_s = scaler.transform(synthetic_data)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(real_s)
        distances_syn, _ = nn.kneighbors(syn_s)
        distances_real, _ = nn.kneighbors(real_s)

        # DCR: Distance to Closest Record
        dcr_syn = np.mean(distances_syn)
        dcr_real = np.mean(distances_real)
        dcr_ratio = dcr_syn / (dcr_real + 1e-10)

        # Privacy score: higher ratio = better privacy
        privacy_score = min(dcr_ratio, 1.0) * 100
        return {
            'dcr_synthetic': float(dcr_syn),
            'dcr_real': float(dcr_real),
            'dcr_ratio': float(dcr_ratio),
            'privacy_score': float(privacy_score),
            'risk_level': 'Low' if privacy_score > 70 else 'Medium' if privacy_score > 40 else 'High'
        }

    def attribute_disclosure_risk(self, real_data, synthetic_data, n_neighbors=5):
        """Measure attribute disclosure risk"""
        scaler = StandardScaler()
        real_s = scaler.fit_transform(real_data)
        syn_s = scaler.transform(synthetic_data)

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(real_s)
        distances, indices = nn.kneighbors(syn_s)

        disclosure_scores = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            neighbors = real_data.iloc[idx]
            syn_record = synthetic_data.iloc[i] if hasattr(synthetic_data, 'iloc') else synthetic_data[i]
            if hasattr(syn_record, 'values'):
                diffs = np.abs(neighbors.values - syn_record.values)
            else:
                diffs = np.abs(neighbors.values - syn_record)
            disclosure_scores.append(np.mean(diffs))

        avg_risk = 1.0 / (np.mean(disclosure_scores) + 1e-10)
        normalized_risk = min(avg_risk / 10, 1.0) * 100
        return {
            'attribute_disclosure_risk': float(normalized_risk),
            'avg_neighbor_distance': float(np.mean(disclosure_scores))
        }

    def statistical_fidelity(self, real_data, synthetic_data):
        """KS test and Wasserstein distance per feature"""
        results = {}
        real_arr = real_data.values if hasattr(real_data, 'values') else real_data
        syn_arr = synthetic_data.values if hasattr(synthetic_data, 'values') else synthetic_data

        ks_stats = []
        wass_dists = []
        for i in range(min(real_arr.shape[1], syn_arr.shape[1])):
            stat, _ = ks_2samp(real_arr[:, i], syn_arr[:, i])
            wd = wasserstein_distance(real_arr[:, i], syn_arr[:, i])
            ks_stats.append(stat)
            wass_dists.append(wd)

        fidelity_score = (1 - np.mean(ks_stats)) * 100
        results['ks_statistic_mean'] = float(np.mean(ks_stats))
        results['wasserstein_distance_mean'] = float(np.mean(wass_dists))
        results['statistical_fidelity_score'] = float(fidelity_score)
        return results

    def correlation_preservation(self, real_data, synthetic_data):
        """Check if synthetic data preserves feature correlations"""
        real_arr = real_data.values if hasattr(real_data, 'values') else real_data
        syn_arr = synthetic_data.values if hasattr(synthetic_data, 'values') else synthetic_data

        real_corr = np.corrcoef(real_arr.T)
        syn_corr = np.corrcoef(syn_arr.T)
        diff = np.abs(real_corr - syn_corr)
        score = (1 - np.mean(diff)) * 100
        return {
            'correlation_preservation_score': float(score),
            'max_correlation_diff': float(np.max(diff))
        }

    def compute_all_metrics(self, real_df, synthetic_df):
        logger.info("Computing all privacy metrics...")
        real_arr = real_df.values if hasattr(real_df, 'values') else real_df
        syn_arr = synthetic_df.values if hasattr(synthetic_df, 'values') else synthetic_df

        real_pd = pd.DataFrame(real_arr, columns=[f'f{i}' for i in range(real_arr.shape[1])])
        syn_pd = pd.DataFrame(syn_arr, columns=[f'f{i}' for i in range(syn_arr.shape[1])])

        mi = self.membership_inference_risk(real_pd, syn_pd)
        ad = self.attribute_disclosure_risk(real_pd, syn_pd)
        sf = self.statistical_fidelity(real_pd, syn_pd)
        cp = self.correlation_preservation(real_pd, syn_pd)

        overall = np.mean([
            mi['privacy_score'],
            100 - ad['attribute_disclosure_risk'],
            sf['statistical_fidelity_score'],
            cp['correlation_preservation_score']
        ])

        self.results = {
            'membership_inference': mi,
            'attribute_disclosure': ad,
            'statistical_fidelity': sf,
            'correlation_preservation': cp,
            'overall_privacy_score': float(overall),
            'grade': 'A' if overall >= 80 else 'B' if overall >= 65 else 'C' if overall >= 50 else 'D'
        }
        return self.results
