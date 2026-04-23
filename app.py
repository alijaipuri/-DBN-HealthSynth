import os
import json
import numpy as np
import pandas as pd
import torch
import threading
import time
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

load_dotenv()

from src.dbn_model import DBN
from src.data_generator import HealthcareDataGenerator
from src.privacy_evaluator import PrivacyEvaluator
from src.ml_evaluator import MLEvaluator
from src.llm_insights import LLMInsights
from src.visualizer import Visualizer

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state
state = {
    'dbn': None,
    'data_gen': None,
    'real_data': None,
    'synthetic_data': None,
    'privacy_results': None,
    'ml_results': None,
    'training_losses': {},
    'fine_tune_losses': [],
    'is_training': False,
    'layer_sizes': [18, 64, 32, 16],
    'n_samples': 500
}

viz = Visualizer()
privacy_eval = PrivacyEvaluator()
ml_eval = MLEvaluator()

try:
    llm = LLMInsights()
except Exception as e:
    print(f"LLM init warning: {e}")
    llm = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    try:
        dg = HealthcareDataGenerator()
        df = dg.load_combined_dataset()
        processed = dg.preprocess(df)
        state['data_gen'] = dg
        state['real_data'] = processed
        stats = {
            'shape': list(df.shape),
            'features': list(df.columns),
            'missing': int(df.isnull().sum().sum()),
            'class_dist': df.iloc[:, -1].value_counts().to_dict() if len(df.columns) > 0 else {}
        }
        sample_records = df.head(5).round(2).to_dict(orient='records')
        return jsonify({'status': 'success', 'stats': stats, 'sample': sample_records})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    if state['is_training']:
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400
    if state['real_data'] is None:
        return jsonify({'status': 'error', 'message': 'Load data first'}), 400

    params = request.json or {}
    pretrain_epochs = int(params.get('pretrain_epochs', 50))
    finetune_epochs = int(params.get('finetune_epochs', 30))
    lr = float(params.get('lr', 0.01))
    batch_size = int(params.get('batch_size', 64))
    layer_config = params.get('layers', [18, 64, 32, 16])

    def train_thread():
        state['is_training'] = True
        try:
            data_tensor = torch.FloatTensor(state['real_data'].values)
            n_features = data_tensor.shape[1]
            layer_config[0] = n_features
            state['layer_sizes'] = layer_config

            dbn = DBN(layer_config, learning_rate=lr)
            state['dbn'] = dbn

            def progress_cb(layer_idx, epoch, total_epochs, loss):
                socketio.emit('training_progress', {
                    'layer': layer_idx + 1,
                    'epoch': epoch + 1,
                    'total_epochs': total_epochs,
                    'loss': round(loss, 6),
                    'phase': 'pretrain'
                })
                time.sleep(0.01)

            socketio.emit('training_status', {'message': 'Pre-training RBM layers...', 'phase': 'pretrain'})
            dbn.pretrain(data_tensor, epochs=pretrain_epochs, batch_size=batch_size,
                         progress_callback=progress_cb)

            socketio.emit('training_status', {'message': 'Fine-tuning with backprop...', 'phase': 'finetune'})
            dbn.fine_tune(data_tensor, epochs=finetune_epochs, batch_size=batch_size)

            state['training_losses'] = {k: v for k, v in dbn.training_losses.items()}
            state['fine_tune_losses'] = dbn.fine_tune_losses

            socketio.emit('training_status', {'message': 'Training complete!', 'phase': 'done'})
            socketio.emit('training_done', {'success': True})
        except Exception as e:
            socketio.emit('training_error', {'message': str(e)})
        finally:
            state['is_training'] = False

    t = threading.Thread(target=train_thread)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started'})

@app.route('/api/generate', methods=['POST'])
def generate_samples():
    if state['dbn'] is None:
        return jsonify({'status': 'error', 'message': 'Train model first'}), 400
    try:
        params = request.json or {}
        n_samples = int(params.get('n_samples', 500))
        temperature = float(params.get('temperature', 1.0))
        noise = float(params.get('noise', 0.1))

        raw = state['dbn'].generate_samples(n_samples, temperature=temperature, noise_level=noise)
        n_features = state['real_data'].shape[1]
        raw = raw[:, :n_features]

        # Inverse transform to original scale
        dg = state['data_gen']
        original_scale = dg.inverse_transform(raw)
        syn_df = pd.DataFrame(original_scale, columns=state['real_data'].columns)
        state['synthetic_data'] = state['real_data'].iloc[:n_samples].copy()
        syn_scaled = pd.DataFrame(raw, columns=state['real_data'].columns)
        state['synthetic_data'] = syn_scaled

        sample_records = syn_df.head(10).round(3).to_dict(orient='records')
        return jsonify({'status': 'success', 'n_generated': n_samples,
                        'sample': sample_records,
                        'columns': list(syn_df.columns)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/evaluate_privacy', methods=['POST'])
def evaluate_privacy():
    if state['synthetic_data'] is None or state['real_data'] is None:
        return jsonify({'status': 'error', 'message': 'Generate data first'}), 400
    try:
        n = min(len(state['real_data']), len(state['synthetic_data']))
        results = privacy_eval.compute_all_metrics(
            state['real_data'].iloc[:n], state['synthetic_data'].iloc[:n])
        state['privacy_results'] = results
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/evaluate_ml', methods=['POST'])
def evaluate_ml():
    if state['synthetic_data'] is None:
        return jsonify({'status': 'error', 'message': 'Generate data first'}), 400
    try:
        results = ml_eval.train_test_utility(
            state['real_data'], state['synthetic_data'], target_col_idx=-1)
        state['ml_results'] = results
        feat_imp = ml_eval.feature_importance_analysis(state['real_data'], target_col_idx=-1)
        return jsonify({'status': 'success', 'results': results, 'feature_importance': feat_imp})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/llm_analysis', methods=['POST'])
def llm_analysis():
    if llm is None:
        return jsonify({'status': 'error', 'message': 'LLM not initialized. Check GROQ_API_KEY'}), 400
    if state['privacy_results'] is None or state['ml_results'] is None:
        return jsonify({'status': 'error', 'message': 'Run evaluations first'}), 400
    try:
        insight = llm.analyze_privacy_report(state['privacy_results'], state['ml_results'])
        arch_explain = llm.explain_dbn_architecture(state['layer_sizes'])
        return jsonify({'status': 'success', 'insight': insight, 'architecture': arch_explain})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/patient_narrative', methods=['POST'])
def patient_narrative():
    if llm is None:
        return jsonify({'status': 'error', 'message': 'LLM not initialized'}), 400
    if state['synthetic_data'] is None:
        return jsonify({'status': 'error', 'message': 'Generate data first'}), 400
    try:
        idx = request.json.get('idx', 0)
        record = state['synthetic_data'].iloc[idx].to_dict()
        # Map back to original scale
        dg = state['data_gen']
        record_arr = np.array(list(record.values())).reshape(1, -1)
        orig = dg.inverse_transform(record_arr)[0]
        orig_record = dict(zip(record.keys(), orig))
        narrative = llm.generate_patient_narrative(orig_record)
        return jsonify({'status': 'success', 'narrative': narrative, 'record': orig_record})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/charts/training', methods=['GET'])
def chart_training():
    if not state['training_losses']:
        return jsonify({'error': 'No training data'}), 400
    chart = viz.training_loss_chart(state['training_losses'], state['fine_tune_losses'])
    return jsonify(chart)

@app.route('/api/charts/correlation', methods=['GET'])
def chart_correlation():
    if state['synthetic_data'] is None:
        return jsonify({'error': 'No synthetic data'}), 400
    n = min(len(state['real_data']), len(state['synthetic_data']))
    chart = viz.correlation_heatmap(state['real_data'].iloc[:n], state['synthetic_data'].iloc[:n])
    return jsonify(chart)

@app.route('/api/charts/distribution', methods=['GET'])
def chart_distribution():
    feature = request.args.get('feature', 'age')
    if state['synthetic_data'] is None:
        return jsonify({'error': 'No synthetic data'}), 400
    n = min(len(state['real_data']), len(state['synthetic_data']))
    chart = viz.distribution_comparison(state['real_data'].iloc[:n], state['synthetic_data'].iloc[:n], feature)
    return jsonify(chart)

@app.route('/api/charts/privacy_radar', methods=['GET'])
def chart_privacy_radar():
    if state['privacy_results'] is None:
        return jsonify({'error': 'No privacy results'}), 400
    chart = viz.privacy_radar(state['privacy_results'])
    return jsonify(chart)

@app.route('/api/charts/ml_comparison', methods=['GET'])
def chart_ml_comparison():
    if state['ml_results'] is None:
        return jsonify({'error': 'No ML results'}), 400
    chart = viz.ml_comparison_chart(state['ml_results'])
    return jsonify(chart)

@app.route('/api/charts/pca', methods=['GET'])
def chart_pca():
    if state['synthetic_data'] is None:
        return jsonify({'error': 'No synthetic data'}), 400
    n = min(len(state['real_data']), len(state['synthetic_data']))
    chart = viz.scatter_pca(state['real_data'].iloc[:n], state['synthetic_data'].iloc[:n])
    return jsonify(chart)

@app.route('/api/export', methods=['GET'])
def export_data():
    if state['synthetic_data'] is None:
        return jsonify({'error': 'No synthetic data'}), 400
    dg = state['data_gen']
    orig = dg.inverse_transform(state['synthetic_data'].values)
    df = pd.DataFrame(orig, columns=state['real_data'].columns)
    return jsonify({
        'status': 'success',
        'data': df.round(3).to_dict(orient='records'),
        'columns': list(df.columns)
    })

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'data_loaded': state['real_data'] is not None,
        'model_trained': state['dbn'] is not None and state['dbn'].is_fitted,
        'synthetic_generated': state['synthetic_data'] is not None,
        'privacy_evaluated': state['privacy_results'] is not None,
        'ml_evaluated': state['ml_results'] is not None,
        'is_training': state['is_training'],
        'layer_sizes': state['layer_sizes']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
