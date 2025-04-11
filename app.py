from flask import Flask, render_template, request, redirect, url_for
import os
from montecarlo import MonteCarloSimulation
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Necessário para gerar gráficos em thread-safe

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Garante que a pasta de imagens existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Processar os dados do formulário
        num_simulations = int(request.form.get('num_simulations', 10000))
        target_duration = float(request.form.get('target_duration', 0)) or None
        
        activities = []
        activity_count = int(request.form.get('activity_count', 0))
        
        for i in range(1, activity_count + 1):
            name = request.form.get(f'activity_{i}_name')
            dist_type = request.form.get(f'activity_{i}_dist')
            
            params = {}
            if dist_type == 'triangular':
                params = {
                    'min': float(request.form.get(f'activity_{i}_min')),
                    'mode': float(request.form.get(f'activity_{i}_mode')),
                    'max': float(request.form.get(f'activity_{i}_max'))
                }
            elif dist_type == 'normal':
                params = {
                    'mean': float(request.form.get(f'activity_{i}_mean')),
                    'std': float(request.form.get(f'activity_{i}_std'))
                }
            elif dist_type == 'lognormal':
                params = {
                    'mu': float(request.form.get(f'activity_{i}_mu')),
                    'sigma': float(request.form.get(f'activity_{i}_sigma'))
                }
            elif dist_type == 'pert':
                params = {
                    'min': float(request.form.get(f'activity_{i}_min')),
                    'mode': float(request.form.get(f'activity_{i}_mode')),
                    'max': float(request.form.get(f'activity_{i}_max'))
                }
            
            activities.append({
                'name': name,
                'dist_type': dist_type,
                'params': params
            })
        
        # Executar simulação
        sim = MonteCarloSimulation(num_simulations=num_simulations)
        for activity in activities:
            sim.add_activity(activity['name'], activity['dist_type'], activity['params'])
        
        results = sim.run_simulation()
        
        # Gerar gráficos e resultados
        stats, graphs = sim.web_analyze_results(target_duration)
        
        return render_template('/results.html', 
                            stats=stats, 
                            graphs=graphs,
                            target_duration=target_duration,
                            activities=activities)
    
    return render_template('/index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)