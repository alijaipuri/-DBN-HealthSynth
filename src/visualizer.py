import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

class Visualizer:
    def correlation_heatmap(self, real_df, synthetic_df):
        real_corr = pd.DataFrame(real_df).corr().round(2)
        syn_corr = pd.DataFrame(synthetic_df).corr().round(2)
        cols = list(real_corr.columns)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Real Data Correlations", "Synthetic Data Correlations"])
        fig.add_trace(go.Heatmap(z=real_corr.values, x=cols, y=cols,
                                  colorscale='RdBu', zmid=0, showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=syn_corr.values, x=cols, y=cols,
                                  colorscale='RdBu', zmid=0), row=1, col=2)
        fig.update_layout(title="Correlation Preservation Analysis",
                          template='plotly_dark', height=500)
        return json.loads(fig.to_json())

    def distribution_comparison(self, real_df, synthetic_df, feature):
        real_arr = pd.DataFrame(real_df)[feature].values if hasattr(real_df, 'columns') else real_df[:, 0]
        syn_arr = pd.DataFrame(synthetic_df)[feature].values if hasattr(synthetic_df, 'columns') else synthetic_df[:, 0]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=real_arr, name='Real', opacity=0.7,
                                    marker_color='#00d4ff', nbinsx=30))
        fig.add_trace(go.Histogram(x=syn_arr, name='Synthetic', opacity=0.7,
                                    marker_color='#ff6b6b', nbinsx=30))
        fig.update_layout(title=f"Distribution: {feature}",
                          barmode='overlay', template='plotly_dark', height=350)
        return json.loads(fig.to_json())

    def training_loss_chart(self, training_losses, fine_tune_losses):
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["RBM Pre-training Losses", "Fine-tuning Loss"])
        colors = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77']
        for i, (name, losses) in enumerate(training_losses.items()):
            fig.add_trace(go.Scatter(y=losses, name=name,
                                     line=dict(color=colors[i % len(colors)], width=2)), row=1, col=1)
        if fine_tune_losses:
            fig.add_trace(go.Scatter(y=fine_tune_losses, name='Fine-tune',
                                     line=dict(color='#c77dff', width=2)), row=1, col=2)
        fig.update_layout(title="DBN Training Progress",
                          template='plotly_dark', height=400)
        return json.loads(fig.to_json())

    def privacy_radar(self, privacy_metrics):
        categories = ['Privacy Score', 'Statistical Fidelity',
                      'Correlation Preservation', 'Attribute Safety']
        mi = privacy_metrics.get('membership_inference', {})
        sf = privacy_metrics.get('statistical_fidelity', {})
        cp = privacy_metrics.get('correlation_preservation', {})
        ad = privacy_metrics.get('attribute_disclosure', {})
        values = [
            privacy_metrics.get('overall_privacy_score', 0),
            sf.get('statistical_fidelity_score', 0),
            cp.get('correlation_preservation_score', 0),
            100 - ad.get('attribute_disclosure_risk', 50)
        ]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.2)',
            line=dict(color='#00d4ff', width=2),
            name='Privacy Profile'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          title="Privacy Metrics Radar",
                          template='plotly_dark', height=400)
        return json.loads(fig.to_json())

    def ml_comparison_chart(self, ml_results):
        models = list(ml_results.get('tstr', {}).keys())
        tstr_acc = [ml_results['tstr'][m]['accuracy'] * 100 for m in models]
        trtr_acc = [ml_results['trtr'][m]['accuracy'] * 100 for m in models]
        fig = go.Figure(data=[
            go.Bar(name='TSTR (Train Synthetic, Test Real)', x=models, y=tstr_acc,
                   marker_color='#00d4ff'),
            go.Bar(name='TRTR (Baseline Real)', x=models, y=trtr_acc,
                   marker_color='#ff6b6b')
        ])
        fig.update_layout(barmode='group', title="Downstream ML Task Performance (TSTR vs TRTR)",
                          yaxis_title="Accuracy (%)", template='plotly_dark', height=400)
        return json.loads(fig.to_json())

    def scatter_pca(self, real_df, synthetic_df):
        from sklearn.decomposition import PCA
        real_arr = real_df.values if hasattr(real_df, 'values') else real_df
        syn_arr = synthetic_df.values if hasattr(synthetic_df, 'values') else synthetic_df
        combined = np.vstack([real_arr, syn_arr])
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(combined)
        n = len(real_arr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=transformed[:n, 0], y=transformed[:n, 1],
                                  mode='markers', name='Real',
                                  marker=dict(color='#00d4ff', size=5, opacity=0.6)))
        fig.add_trace(go.Scatter(x=transformed[n:, 0], y=transformed[n:, 1],
                                  mode='markers', name='Synthetic',
                                  marker=dict(color='#ff6b6b', size=5, opacity=0.6)))
        fig.update_layout(title="PCA: Real vs Synthetic Distribution",
                          xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                          yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                          template='plotly_dark', height=400)
        return json.loads(fig.to_json())
