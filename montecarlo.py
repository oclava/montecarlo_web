import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import triang, norm, lognorm, beta
import base64
import io

class MonteCarloSimulation:
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations
        self.results = None
        self.activities = {}
        
    def add_activity(self, name, distribution, params):
        self.activities[name] = {
            'distribution': distribution,
            'params': params
        }
    
    def run_simulation(self):
        results_df = pd.DataFrame(index=range(self.num_simulations))
        
        for activity, config in self.activities.items():
            dist = config['distribution']
            params = config['params']
            
            if dist == 'triangular':
                left = params['min']
                mode = params['mode']
                right = params['max']
                scale = right - left
                c = (mode - left) / scale
                results_df[activity] = triang.rvs(c, loc=left, scale=scale, size=self.num_simulations)
                
            elif dist == 'normal':
                results_df[activity] = norm.rvs(loc=params['mean'], scale=params['std'], size=self.num_simulations)
                
            elif dist == 'lognormal':
                results_df[activity] = lognorm.rvs(s=params['sigma'], scale=np.exp(params['mu']), size=self.num_simulations)
                
            elif dist == 'pert':
                min_ = params['min']
                mode = params['mode']
                max_ = params['max']
                mean = (min_ + 4 * mode + max_) / 6
                std = (max_ - min_) / 6
                alpha = ((mean - min_) / (max_ - min_)) * (((mean - min_) * (max_ - mean)) / std**2 - 1)
                beta_ = ((max_ - mean) / (mean - min_)) * alpha
                results_df[activity] = min_ + (max_ - min_) * beta.rvs(alpha, beta_, size=self.num_simulations)
        
        results_df['total_duration'] = results_df.sum(axis=1)
        self.results = results_df
        return results_df
    
    def web_analyze_results(self, target_duration=None):
        if self.results is None:
            return None, None
        
        total_duration = self.results['total_duration']
        
        # Estatísticas
        stats = {
            'mean': np.mean(total_duration),
            'median': np.median(total_duration),
            'std': np.std(total_duration),
            'min': np.min(total_duration),
            'max': np.max(total_duration),
            'percentiles': {p: np.percentile(total_duration, p) for p in [10, 50, 80, 90, 95]},
            'probability': np.mean(total_duration <= target_duration) * 100 if target_duration else None,
            'correlations': self.results.corr()['total_duration'].drop('total_duration').sort_values(ascending=False).to_dict()
        }
        
        # Gráficos
        graphs = {
            'histogram': self._plot_to_base64(self._plot_histogram, target_duration),
            'boxplot': self._plot_to_base64(self._plot_boxplot),
            'correlation': self._plot_to_base64(self._plot_correlation),
            'cumulative': self._plot_to_base64(self._plot_cumulative, target_duration)
        }
        
        return stats, graphs
    
    def _plot_to_base64(self, plot_function, *args):
        plt.figure()
        plot_function(*args)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def _plot_histogram(self, target_duration):
        sns.histplot(self.results['total_duration'], kde=True)
        if target_duration:
            plt.axvline(x=target_duration, color='r', linestyle='--', label='Prazo alvo')
        plt.title('Distribuição da Duração Total do Projeto')
        plt.xlabel('Dias')
        plt.ylabel('Frequência')
        plt.legend()
    
    def _plot_boxplot(self):
        activities_df = self.results.drop(columns=['total_duration'])
        sns.boxplot(data=activities_df)
        plt.title('Distribuição das Durações das Atividades')
        plt.ylabel('Dias')
        plt.xticks(rotation=45)
    
    def _plot_correlation(self):
        corr = self.results.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                   annot_kws={"size": 8}, vmin=-1, vmax=1)
        plt.title('Matriz de Correlação')
    
    def _plot_cumulative(self, target_duration):
        sorted_durations = np.sort(self.results['total_duration'])
        cum_prob = np.arange(1, len(sorted_durations)+1) / len(sorted_durations)
        plt.plot(sorted_durations, cum_prob*100)
        plt.axhline(y=80, color='g', linestyle='--', label='80% de confiança')
        if target_duration:
            plt.axvline(x=target_duration, color='r', linestyle='--', label='Prazo alvo')
        plt.title('Probabilidade Cumulativa de Conclusão')
        plt.xlabel('Dias')
        plt.ylabel('Probabilidade de Conclusão (%)')
        plt.legend()