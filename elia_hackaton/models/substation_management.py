import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
import networkx as nx
from sklearn.cluster import KMeans
import seaborn as sns


class SubstationManager:
    def __init__(self, models_dir='saved_models', results_dir='results', prediction_dir='test'):
        """
        Initialize the Substation Management System
        
        Parameters:
        -----------
        models_dir : str
            Directory containing saved transformer models
        results_dir : str
            Directory containing transformer data
        prediction_dir : str
            Directory containing prediction results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.prediction_dir = prediction_dir
        self.transformers = {}
        self.network = None
        self.temperature_limits = {}
        self.default_temp_limit = 80  # Default transformer temperature limit in Celsius
        self.load_factors = {}
        self.substation_clusters = {}
        self.reinforcement_priorities = []

        # Brussels area transformer data
        self.brussels_transformers = {
            '24917': {'city': 'Mechelen'},
            '33291': {'city': 'Romsee'},
            '76075': {'city': 'Uccle'},
            '76632': {'city': 'Uccle'},
            '5004597': {'city': 'Auvelais'},
            '5004625': {'city': 'Auvelais'},
            '5017843': {'city': 'Nieuwpoort'},
            '5017873': {'city': 'Nieuwpoort'},
            '5025312': {'city': 'Molenbeek-saint-jean'},
            '5026188': {'city': 'Romsee'},
            '5027447': {'city': 'Lens'},
            '5030256': {'city': 'Schaerbeek'},
            '50033827': {'city': 'Wilsele'},
            '50249457': {'city': 'Anderlecht'},
            '50250035': {'city': 'Nivelles'},
            '50277055': {'city': 'Montignies-sur-sambre'},
            '50282127': {'city': 'Farciennes'}
        }

        # Create city groups for clustering
        self.city_groups = {}
        for transformer_id, data in self.brussels_transformers.items():
            city = data['city']
            if city not in self.city_groups:
                self.city_groups[city] = []
            self.city_groups[city].append(transformer_id)

        # Geographic coordinates for cities (approximate, for visualization)
        # These are simplified coordinates for visualization purposes
        self.city_coordinates = {
            'Mechelen': (51.0259, 4.4776),
            'Romsee': (50.6371, 5.6337),
            'Uccle': (50.8003, 4.3371),
            'Auvelais': (50.4355, 4.6225),
            'Nieuwpoort': (51.1289, 2.7499),
            'Molenbeek-saint-jean': (50.8583, 4.3306),
            'Lens': (50.5565, 3.9385),
            'Schaerbeek': (50.8676, 4.3826),
            'Wilsele': (50.8962, 4.7019),
            'Anderlecht': (50.8333, 4.3167),
            'Nivelles': (50.5981, 4.3302),
            'Montignies-sur-sambre': (50.4042, 4.4662),
            'Farciennes': (50.4299, 4.5305)
        }

    def load_transformer_data(self):
        """Load transformer data and temperature predictions"""
        print("Loading transformer data...")

        # Load all available transformers and their data
        for filename in os.listdir(self.results_dir):
            if filename.endswith(".csv"):
                equipment_id = filename.split('.csv')[0]
                file_path = os.path.join(self.results_dir, filename)

                # Load transformer data
                self.transformers[equipment_id] = {
                    'data': pd.read_csv(file_path),
                    'model_available': os.path.exists(os.path.join(self.models_dir, f'model_{equipment_id}.pth')),
                    'prediction_available': os.path.exists(os.path.join(self.prediction_dir, f'{equipment_id}.csv'))
                }

                # Add Brussels transformer metadata if available
                if equipment_id in self.brussels_transformers:
                    self.transformers[equipment_id].update(self.brussels_transformers[equipment_id])

                # Set default temperature limit
                self.temperature_limits[equipment_id] = self.default_temp_limit

                # If transformer model parameters are available, load them
                params_path = os.path.join(self.models_dir, f'params_{equipment_id}.json')
                if os.path.exists(params_path):
                    with open(params_path, 'r') as f:
                        self.transformers[equipment_id]['params'] = json.load(f)

                # Load predictions if available
                if self.transformers[equipment_id]['prediction_available']:
                    pred_path = os.path.join(self.prediction_dir, f'{equipment_id}.csv')
                    self.transformers[equipment_id]['predictions'] = pd.read_csv(pred_path)

        # If no data files found, initialize with Brussels transformer data
        if not self.transformers:
            print("No data files found. Initializing with Brussels transformer metadata.")
            for equipment_id, data in self.brussels_transformers.items():
                self.transformers[equipment_id] = {
                    'city': data['city'],
                    'data': pd.DataFrame({'K': np.random.normal(0.7, 0.1, 100),
                                          'temperature': np.random.normal(60, 10, 100),
                                          'dateTime': pd.date_range(start='2024-01-01', periods=100, freq='H')}),
                    'model_available': False,
                    'prediction_available': False
                }
                # Set default temperature limit
                self.temperature_limits[equipment_id] = self.default_temp_limit

        print(f"Loaded data for {len(self.transformers)} transformers")
        return self.transformers

    def analyze_transformer_capacity(self):
        """Analyze transformer capacity and calculate load factors"""
        print("Analyzing transformer capacity...")

        for transformer_id, transformer in self.transformers.items():
            data = transformer['data']

            # Calculate historical max load and temperature
            max_load = data['K'].max()
            max_temp = data['temperature'].max()
            avg_load = data['K'].mean()

            # Calculate standard deviation of load
            load_std = data['K'].std()

            # Calculate how often transformer operates near its thermal limit
            high_temp_threshold = self.temperature_limits[transformer_id] * 0.9  # 90% of limit
            high_temp_percentage = (data['temperature'] > high_temp_threshold).mean() * 100

            # Calculate typical load to temperature relationship at high loads
            high_load_mask = data['K'] > np.percentile(data['K'], 75)
            if high_load_mask.sum() > 0:
                high_load_temp_ratio = data.loc[high_load_mask, 'temperature'].mean() / data.loc[
                    high_load_mask, 'K'].mean()
            else:
                high_load_temp_ratio = 0

            # Calculate load headroom (estimated additional load capacity)
            temp_headroom = self.temperature_limits[transformer_id] - max_temp
            if high_load_temp_ratio > 0:
                load_headroom = temp_headroom / high_load_temp_ratio
            else:
                load_headroom = 0

            # Calculate load factor: higher values indicate higher priority for reinforcement
            # Formula considers current load, headroom, variability, and thermal stress
            load_factor = (0.5 * avg_load / max_load) + (0.2 * (1 - load_headroom / max_load)) + \
                          (0.15 * load_std / max_load) + (0.15 * high_temp_percentage / 100)

            # Store results
            self.transformers[transformer_id]['metrics'] = {
                'max_load': max_load,
                'avg_load': avg_load,
                'max_temp': max_temp,
                'load_std': load_std,
                'high_temp_percentage': high_temp_percentage,
                'load_headroom': load_headroom,
                'load_factor': load_factor
            }

            self.load_factors[transformer_id] = load_factor

        return self.load_factors

    def build_network_model(self):
        """Build a network model of the substations based on city locations"""
        # Create a network graph
        self.network = nx.Graph()

        # Add nodes (transformers)
        for transformer_id, transformer in self.transformers.items():
            # Check if metrics exist, if not use defaults
            if 'metrics' not in transformer:
                transformer['metrics'] = {
                    'max_load': 1.0,
                    'load_factor': 0.7
                }

            self.network.add_node(transformer_id,
                                  load_factor=self.load_factors.get(transformer_id, 0.7),
                                  capacity=transformer['metrics']['max_load'],
                                  city=transformer.get('city', 'Unknown'))

        # Create connections based on geographic proximity and same city
        transformer_ids = list(self.transformers.keys())

        # Connect transformers in the same city
        for city, transformers in self.city_groups.items():
            if len(transformers) > 1:
                for i, id1 in enumerate(transformers):
                    for id2 in transformers[i + 1:]:
                        if id1 in self.network.nodes() and id2 in self.network.nodes():
                            self.network.add_edge(id1, id2, weight=0.5, edge_type='same_city')

        # Connect cities that are geographically close
        # Create a list of cities with their coordinates
        cities = list(self.city_coordinates.keys())

        # Compute distances and connect nearby cities
        connected_cities = set()
        for i, city1 in enumerate(cities):
            for city2 in cities[i + 1:]:
                # Calculate Euclidean distance between cities (simplified)
                x1, y1 = self.city_coordinates[city1]
                x2, y2 = self.city_coordinates[city2]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

                # Connect cities if they're relatively close (threshold can be adjusted)
                if distance < 0.5:  # This is a simplified threshold
                    connected_cities.add((city1, city2))

        # Now connect transformers between connected cities
        for city1, city2 in connected_cities:
            transformers1 = self.city_groups.get(city1, [])
            transformers2 = self.city_groups.get(city2, [])

            # Connect one transformer from each city (to avoid too many edges)
            if transformers1 and transformers2:
                id1 = transformers1[0]
                id2 = transformers2[0]
                if id1 in self.network.nodes() and id2 in self.network.nodes():
                    self.network.add_edge(id1, id2, weight=1.0, edge_type='nearby_city')

        return self.network

    def identify_transformer_clusters(self, n_clusters=5):
        """Group transformers into logical clusters based on location and metrics"""

        # --- Method 1: Clustering by city location ---
        # Create city-based clusters
        city_clusters = {}
        cluster_id = 0

        # First, assign each city to a cluster
        for city in self.city_groups.keys():
            city_clusters[city] = cluster_id
            cluster_id += 1

        # Compute connected_cities based on geographic proximity
        cities = list(self.city_coordinates.keys())
        connected_cities = set()
        for i, city1 in enumerate(cities):
            for city2 in cities[i + 1:]:
                x1, y1 = self.city_coordinates[city1]
                x2, y2 = self.city_coordinates[city2]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if distance < 0.5:  # Simplified threshold
                    connected_cities.add((city1, city2))

        # Merge similar clusters based on geographic proximity
        for (city1, city2) in connected_cities:
            if city1 in city_clusters and city2 in city_clusters:
                if city_clusters[city1] != city_clusters[city2]:
                    old_cluster = city_clusters[city2]
                    new_cluster = city_clusters[city1]
                    # Update all cities with the old cluster to the new one
                    for city, cluster in city_clusters.items():
                        if cluster == old_cluster:
                            city_clusters[city] = new_cluster

        # Assign transformers to clusters based on their city
        for transformer_id, transformer in self.transformers.items():
            if 'city' in transformer:
                city = transformer['city']
                if city in city_clusters:
                    self.substation_clusters[transformer_id] = city_clusters[city]
                else:
                    # Assign to a new cluster if city not found
                    self.substation_clusters[transformer_id] = len(city_clusters)
                    city_clusters[city] = len(city_clusters)

        # --- Method 2: Use K-means as fallback or additional method ---
        transformer_features = []
        transformer_ids = []
        use_kmeans = False

        for transformer_id, transformer in self.transformers.items():
            if 'metrics' in transformer:
                metrics = transformer['metrics']
                features = [
                    metrics['max_load'],
                    metrics['max_temp'],
                    metrics['load_factor'],
                    metrics['load_headroom']
                ]
                transformer_features.append(features)
                transformer_ids.append(transformer_id)
                use_kmeans = True

        if use_kmeans and len(transformer_features) >= n_clusters:
            features_array = np.array(transformer_features)
            features_normalized = (features_array - features_array.mean(axis=0)) / features_array.std(axis=0)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_normalized)
            kmeans_clusters = {transformer_ids[i]: int(clusters[i]) for i in range(len(transformer_ids))}
            for transformer_id in transformer_ids:
                if transformer_id not in self.substation_clusters:
                    self.substation_clusters[transformer_id] = kmeans_clusters[transformer_id]

        # Calculate cluster properties (optional)
        cluster_properties = {}
        for cluster_id in set(self.substation_clusters.values()):
            cluster_transformers = [tid for tid, cid in self.substation_clusters.items() if cid == cluster_id]
            cluster_cities = []
            for tid in cluster_transformers:
                if tid in self.transformers and 'city' in self.transformers[tid]:
                    city = self.transformers[tid]['city']
                    if city not in cluster_cities:
                        cluster_cities.append(city)
            load_factors = [self.load_factors.get(tid, 0.5) for tid in cluster_transformers]
            avg_load_factor = np.mean(load_factors) if load_factors else 0.5
            cluster_properties[cluster_id] = {
                'transformers': cluster_transformers,
                'cities': cluster_cities,
                'avg_load_factor': avg_load_factor,
                'count': len(cluster_transformers)
            }

        return cluster_properties

    def visualize_network_by_city(self):
        """Visualize the substation network grouped by city"""
        if self.network is None:
            self.build_network_model()

        plt.figure(figsize=(14, 10))

        # Create city-to-color mapping
        cities = list(set(self.transformers[tid].get('city', 'Unknown') for tid in self.transformers))
        city_colors = {}
        colormap = plt.cm.rainbow
        for i, city in enumerate(cities):
            city_colors[city] = colormap(i / len(cities))

        # Create position layout based on city coordinates
        pos = {}
        for node in self.network.nodes():
            city = self.transformers[node].get('city', 'Unknown')

            # Add small random offset for nodes in the same city
            if city in self.city_coordinates:
                x, y = self.city_coordinates[city]
                # Add small jitter to avoid overlapping nodes
                x_jitter = np.random.normal(0, 0.01)
                y_jitter = np.random.normal(0, 0.01)
                pos[node] = (x + x_jitter, y + y_jitter)
            else:
                # Random position if no coordinates
                pos[node] = (np.random.random(), np.random.random())

        # Draw nodes with colors based on city
        for city in cities:
            # Get nodes for this city
            city_nodes = [node for node in self.network.nodes() if self.transformers[node].get('city', '') == city]

            # Get node sizes based on capacity or use default
            node_sizes = []
            for node in city_nodes:
                if 'metrics' in self.transformers[node]:
                    size = 300 * self.transformers[node]['metrics']['max_load'] / max(
                        t['metrics']['max_load'] for t in self.transformers.values() if 'metrics' in t
                    ) if any('metrics' in t for t in self.transformers.values()) else 300
                else:
                    size = 300
                node_sizes.append(size)

            # Draw nodes for this city
            nx.draw_networkx_nodes(
                self.network, pos,
                nodelist=city_nodes,
                node_color=[city_colors[city]] * len(city_nodes),
                node_size=node_sizes,
                label=city
            )

        # Draw edges with different styles for same city vs inter-city
        same_city_edges = [(u, v) for u, v, d in self.network.edges(data=True) if d.get('edge_type') == 'same_city']
        nearby_city_edges = [(u, v) for u, v, d in self.network.edges(data=True) if d.get('edge_type') == 'nearby_city']

        nx.draw_networkx_edges(self.network, pos, edgelist=same_city_edges, alpha=0.5, style='solid')
        nx.draw_networkx_edges(self.network, pos, edgelist=nearby_city_edges, alpha=0.7, style='dashed', width=2)

        # Add labels
        nx.draw_networkx_labels(self.network, pos, font_size=8)

        plt.title('Brussels Region Substation Network by City')
        plt.axis('off')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('brussels_substation_network.png')
        plt.close()

        return 'brussels_substation_network.png'

    def prioritize_reinforcements(self):
        """Determine which substations to prioritize for reinforcements"""
        # Create a priority score based on multiple factors
        priority_scores = {}

        for transformer_id, transformer in self.transformers.items():
            if 'metrics' not in transformer:
                continue

            metrics = transformer['metrics']

            # Calculate priority score based on:
            # 1. Load factor (higher is worse)
            # 2. Headroom (lower is worse)
            # 3. High temperature percentage (higher is worse)
            # 4. Network centrality (higher is more important)

            # Network centrality (how important is this node in the grid)
            centrality = 0.5  # Default
            if self.network is not None:
                centrality = nx.degree_centrality(self.network).get(transformer_id, 0.5)

            # Priority score calculation
            priority_score = (0.4 * metrics['load_factor']) + \
                             (0.3 * (1 - metrics['load_headroom'] / metrics['max_load'])) + \
                             (0.2 * metrics['high_temp_percentage'] / 100) + \
                             (0.1 * centrality)

            priority_scores[transformer_id] = priority_score

        # Sort transformers by priority score (descending)
        self.reinforcement_priorities = sorted(
            priority_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return self.reinforcement_priorities

    def visualize_clusters(self):
        """Create a visualization of transformer clusters"""
        if not self.substation_clusters:
            self.identify_transformer_clusters()

        if self.network is None:
            self.build_network_model()

        plt.figure(figsize=(14, 10))

        # Create position layout based on city coordinates
        pos = {}
        for node in self.network.nodes():
            city = self.transformers[node].get('city', 'Unknown')

            # Add small random offset for nodes in the same city
            if city in self.city_coordinates:
                x, y = self.city_coordinates[city]
                # Add small jitter to avoid overlapping nodes
                x_jitter = np.random.normal(0, 0.01)
                y_jitter = np.random.normal(0, 0.01)
                pos[node] = (x + x_jitter, y + y_jitter)
            else:
                # Random position if no coordinates
                pos[node] = (np.random.random(), np.random.random())

        # Create a colormap for clusters
        num_clusters = max(self.substation_clusters.values()) + 1
        colormap = plt.cm.viridis

        # Draw nodes with colors based on cluster
        for cluster_id in range(num_clusters):
            # Get nodes for this cluster
            cluster_nodes = [node for node, c in self.substation_clusters.items() if c == cluster_id]

            # Get node sizes based on capacity or use default
            node_sizes = []
            for node in cluster_nodes:
                if 'metrics' in self.transformers[node]:
                    size = 300 * self.transformers[node]['metrics']['max_load'] / max(
                        t['metrics']['max_load'] for t in self.transformers.values() if 'metrics' in t
                    ) if any('metrics' in t for t in self.transformers.values()) else 300
                else:
                    size = 300
                node_sizes.append(size)

            # Get city names for this cluster for the label
            cities = set()
            for node in cluster_nodes:
                if node in self.transformers and 'city' in self.transformers[node]:
                    cities.add(self.transformers[node]['city'])
            city_str = ", ".join(sorted(cities)) if cities else f"Cluster {cluster_id}"

            # Draw nodes for this cluster
            nx.draw_networkx_nodes(
                self.network, pos,
                nodelist=cluster_nodes,
                node_color=[colormap(cluster_id / num_clusters)] * len(cluster_nodes),
                node_size=node_sizes,
                label=f"Cluster {cluster_id}: {city_str}"
            )

        # Draw edges
        nx.draw_networkx_edges(self.network, pos, alpha=0.2)

        # Add labels
        nx.draw_networkx_labels(self.network, pos, font_size=8)

        # Add city labels
        city_centers = {}
        for city, coords in self.city_coordinates.items():
            city_centers[city] = coords

        for city, (x, y) in city_centers.items():
            plt.text(x, y, city, fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.title('Brussels Region Transformer Clusters')
        plt.axis('off')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig('brussels_transformer_clusters.png')
        plt.close()

        return 'brussels_transformer_clusters.png'

    def calculate_safe_overload_limits(self):
        """Calculate safe overload limits for each transformer"""
        overload_limits = {}

        for transformer_id, transformer in self.transformers.items():
            if 'metrics' not in transformer:
                continue

            metrics = transformer['metrics']
            max_load = metrics['max_load']

            # Base overload capacity on temperature predictions and headroom
            # Start with current max load
            base_overload_limit = max_load

            # Add headroom based on temperature predictions
            if metrics['load_headroom'] > 0:
                # Conservative approach: use 80% of calculated headroom
                overload_capacity = max_load + (metrics['load_headroom'] * 0.8)
            else:
                # If no headroom, limit to current max load
                overload_capacity = max_load

            # Adjust based on ambient temperature patterns
            data = transformer['data']
            seasonal_factor = 1.0

            # Calculate seasonal adjustment (allows more overload in cooler conditions)
            if 'dateTime' in data and 'temperature' in data:
                data['dateTime'] = pd.to_datetime(data['dateTime'])
                data['month'] = data['dateTime'].dt.month

                # Get average ambient temperature by month
                monthly_temp = data.groupby('month')['temperature'].mean()

                # Calculate seasonal factor (higher in winter/cooler months)
                if not monthly_temp.empty:
                    temp_range = monthly_temp.max() - monthly_temp.min()
                    if temp_range > 0:
                        # Scale factor between 0.9 (hot months) and 1.1 (cold months)
                        month_factors = 1.1 - (0.2 * (monthly_temp - monthly_temp.min()) / temp_range)
                        seasonal_factor = month_factors.mean()

            # Final overload limit with seasonal adjustment
            safe_overload_limit = overload_capacity * seasonal_factor

            # Don't allow more than 30% overload for safety
            max_allowed_overload = max_load * 1.3
            safe_overload_limit = min(safe_overload_limit, max_allowed_overload)

            overload_limits[transformer_id] = safe_overload_limit

        return overload_limits

    def create_reinforcement_report_by_city(self):
        """Create a reinforcement report grouped by city"""
        if not self.reinforcement_priorities:
            self.prioritize_reinforcements()

        # Create a dataframe with reinforcement priorities grouped by city
        priority_data = []

        for transformer_id, score in self.reinforcement_priorities:
            if transformer_id in self.transformers and 'metrics' in self.transformers[transformer_id]:
                metrics = self.transformers[transformer_id]['metrics']
                city = self.transformers[transformer_id].get('city', 'Unknown')

                priority_data.append({
                    'transformer_id': transformer_id,
                    'city': city,
                    'priority_score': score,
                    'load_factor': metrics['load_factor'],
                    'load_headroom': metrics['load_headroom'],
                    'max_temperature': metrics['max_temp'],
                    'high_temp_percentage': metrics['high_temp_percentage'],
                    'cluster': self.substation_clusters.get(transformer_id, -1)
                })

        if priority_data:
            priority_df = pd.DataFrame(priority_data)

            # Plot priority scores by city
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='city', y='priority_score', data=priority_df, palette='viridis')
            plt.title('Transformer Reinforcement Priority Scores by City')
            plt.xlabel('City')
            plt.ylabel('Priority Score')
            plt.xticks(rotation=45, ha='right')

            # Add transformer IDs as labels
            for i, city in enumerate(priority_df['city'].unique()):
                city_transformers = priority_df[priority_df['city'] == city]['transformer_id'].tolist()
                city_scores = priority_df[priority_df['city'] == city]['priority_score'].tolist()

                for j, (tid, score) in enumerate(zip(city_transformers, city_scores)):
                    ax.text(i, score + 0.01, tid, ha='center', va='bottom', rotation=90, fontsize=8)

            plt.tight_layout()
            plt.savefig('brussels_reinforcement_priorities.png')
            plt.close()

            # Create a heatmap of priority metrics by city
            city_metrics = priority_df.groupby('city').agg({
                'priority_score': 'mean',
                'load_factor': 'mean',
                'load_headroom': 'mean',
                'max_temperature': 'mean',
                'high_temp_percentage': 'mean'
            }).reset_index()

            plt.figure(figsize=(12, 8))
            metrics_to_plot = ['priority_score', 'load_factor', 'load_headroom', 'max_temperature',
                               'high_temp_percentage']

            # Normalize the data for better visualization
            for col in metrics_to_plot:
                if col != 'city':
                    city_metrics[col + '_norm'] = (city_metrics[col] - city_metrics[col].min()) / (
                                city_metrics[col].max() - city_metrics[col].min())

            norm_columns = [col + '_norm' for col in metrics_to_plot]
            heatmap_data = city_metrics.pivot_table(index='city', values=norm_columns)

            # Rename columns for better readability
            heatmap_data.columns = ['Priority', 'Load Factor', 'Load Headroom', 'Max Temp', 'High Temp %']

            sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.2f')
            plt.title('Normalized Priority Metrics by City')
            plt.tight_layout()
            plt.savefig('brussels_city_priority_heatmap.png')
            plt.close()

            # Save priority list
            # Save priority list to CSV
            priority_df.to_csv('brussels_reinforcement_priorities.csv', index=False)

            return {
                'priority_df': priority_df,
                'city_metrics': city_metrics,
                'charts': [
                    'brussels_reinforcement_priorities.png',
                    'brussels_city_priority_heatmap.png'
                ]
            }
        else:
            print("No priority data available")
            return None

    def predict_future_load_patterns(self, days_ahead=30):
        """Predict future load patterns based on historical data"""
        future_predictions = {}

        for transformer_id, transformer in self.transformers.items():
            if 'data' not in transformer:
                continue

            data = transformer['data']

            # Check if we have datetime and load data
            if 'dateTime' in data.columns and 'K' in data.columns:
                # Convert to datetime if it's not already
                data['dateTime'] = pd.to_datetime(data['dateTime'])

                # Get the last date in our dataset
                if not data.empty:
                    last_date = data['dateTime'].max()

                    # Create a date range for future predictions
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=days_ahead,
                        freq='D'
                    )

                    # Create a simple prediction based on historical patterns
                    # This is a placeholder for more sophisticated forecasting methods

                    # Method 1: Use seasonal patterns if available
                    if len(data) > 30:  # Only if we have enough historical data
                        # Extract day of week and time features
                        data['day_of_week'] = data['dateTime'].dt.dayofweek
                        data['month'] = data['dateTime'].dt.month

                        # Calculate average load by day of week and month
                        daily_pattern = data.groupby('day_of_week')['K'].mean()
                        monthly_pattern = data.groupby('month')['K'].mean()

                        # Create predictions
                        future_loads = []
                        for date in future_dates:
                            day_of_week = date.dayofweek
                            month = date.month

                            # Combine daily and monthly patterns
                            predicted_load = (
                                    daily_pattern.get(day_of_week, data['K'].mean()) * 0.6 +
                                    monthly_pattern.get(month, data['K'].mean()) * 0.4
                            )

                            future_loads.append(predicted_load)
                    else:
                        # Fallback: use average load for predictions
                        avg_load = data['K'].mean()
                        future_loads = [avg_load] * len(future_dates)

                    # Store predictions
                    future_predictions[transformer_id] = pd.DataFrame({
                        'dateTime': future_dates,
                        'predicted_load': future_loads
                    })

        return future_predictions

    def simulate_grid_stress_scenarios(self, n_scenarios=3):
        """Simulate grid stress scenarios to test robustness"""
        if self.network is None:
            self.build_network_model()

        # Get overload limits
        overload_limits = self.calculate_safe_overload_limits()

        # Define stress scenarios
        scenarios = [
            {
                'name': 'Summer Heat Wave',
                'load_multiplier': 1.2,  # 20% increase in load due to cooling
                'affected_cities': ['Uccle', 'Molenbeek-saint-jean', 'Schaerbeek', 'Anderlecht'],
                'stress_type': 'temperature'
            },
            {
                'name': 'Winter Peak',
                'load_multiplier': 1.3,  # 30% increase in load due to heating
                'affected_cities': ['Mechelen', 'Lens', 'Wilsele', 'Nivelles'],
                'stress_type': 'demand'
            },
            {
                'name': 'Transformer Failure',
                'load_multiplier': 0,  # Failed transformer
                'failed_transformers': ['5030256', '50033827'],  # Random transformers
                'stress_type': 'failure'
            }
        ]

        # Calculate original total grid capacity
        original_capacity = sum(
            transformer['metrics']['max_load']
            for transformer in self.transformers.values()
            if 'metrics' in transformer
        )

        # Run simulations
        simulation_results = []

        for scenario in scenarios:
            # Copy the network for simulation
            sim_network = self.network.copy()

            # Apply scenario effects
            if scenario['stress_type'] in ['temperature', 'demand']:
                # Increased load in specific cities
                for transformer_id in self.transformers:
                    if transformer_id in sim_network.nodes():
                        city = self.transformers[transformer_id].get('city', '')

                        if city in scenario['affected_cities']:
                            # Increase load
                            current_load = self.transformers[transformer_id]['metrics']['max_load'] \
                                if 'metrics' in self.transformers[transformer_id] else 1.0

                            new_load = current_load * scenario['load_multiplier']
                            sim_network.nodes[transformer_id]['simulated_load'] = new_load

                            # Check if exceeds limit
                            overload_limit = overload_limits.get(transformer_id, current_load * 1.2)
                            sim_network.nodes[transformer_id]['overloaded'] = new_load > overload_limit
                        else:
                            # Unaffected transformers
                            current_load = self.transformers[transformer_id]['metrics']['max_load'] \
                                if 'metrics' in self.transformers[transformer_id] else 1.0
                            sim_network.nodes[transformer_id]['simulated_load'] = current_load
                            sim_network.nodes[transformer_id]['overloaded'] = False

            elif scenario['stress_type'] == 'failure':
                # Simulate transformer failures
                for transformer_id in scenario['failed_transformers']:
                    if transformer_id in sim_network.nodes():
                        # Mark as failed
                        sim_network.nodes[transformer_id]['failed'] = True
                        sim_network.nodes[transformer_id]['simulated_load'] = 0

                        # Redistribute load to connected transformers
                        neighbors = list(sim_network.neighbors(transformer_id))
                        failed_load = self.transformers[transformer_id]['metrics']['max_load'] \
                            if 'metrics' in self.transformers[transformer_id] else 1.0

                        if neighbors:
                            load_per_neighbor = failed_load / len(neighbors)

                            for neighbor in neighbors:
                                # Increase neighbor's load
                                current_load = self.transformers[neighbor]['metrics']['max_load'] \
                                    if 'metrics' in self.transformers[neighbor] else 1.0

                                new_load = current_load + load_per_neighbor
                                sim_network.nodes[neighbor]['simulated_load'] = new_load

                                # Check if exceeds limit
                                overload_limit = overload_limits.get(neighbor, current_load * 1.2)
                                sim_network.nodes[neighbor]['overloaded'] = new_load > overload_limit

            # Calculate scenario metrics
            overloaded_transformers = [
                node for node in sim_network.nodes()
                if sim_network.nodes[node].get('overloaded', False)
            ]

            failed_transformers = [
                node for node in sim_network.nodes()
                if sim_network.nodes[node].get('failed', False)
            ]

            # Calculate remaining capacity
            remaining_capacity = sum(
                sim_network.nodes[node].get('simulated_load', 0)
                for node in sim_network.nodes()
                if not sim_network.nodes[node].get('failed', False)
            )

            capacity_percentage = (remaining_capacity / original_capacity) * 100 if original_capacity > 0 else 0

            # Store results
            simulation_results.append({
                'scenario': scenario['name'],
                'stress_type': scenario['stress_type'],
                'overloaded_transformers': overloaded_transformers,
                'failed_transformers': failed_transformers,
                'capacity_percentage': capacity_percentage,
                'affected_cities': scenario.get('affected_cities', []),
                'network': sim_network
            })

        # Create visualization
        plt.figure(figsize=(12, 8))

        # Bar chart of capacity percentages
        scenarios = [r['scenario'] for r in simulation_results]
        capacities = [r['capacity_percentage'] for r in simulation_results]

        bars = plt.bar(scenarios, capacities, color='skyblue')

        # Add overloaded transformer counts
        for i, result in enumerate(simulation_results):
            overload_count = len(result['overloaded_transformers'])
            plt.text(i, capacities[i] + 2, f"Overloaded: {overload_count}",
                     ha='center', va='bottom', color='red')

        plt.axhline(y=100, color='green', linestyle='--', label='Original Capacity')
        plt.axhline(y=80, color='orange', linestyle='--', label='Critical Threshold (80%)')

        plt.title('Grid Stress Scenario Analysis')
        plt.ylabel('Remaining Capacity (%)')
        plt.ylim(0, 120)
        plt.legend()
        plt.tight_layout()
        plt.savefig('brussels_stress_scenarios.png')
        plt.close()

        return {
            'results': simulation_results,
            'visualization': 'brussels_stress_scenarios.png'
        }

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        # Ensure we have analyzed all data
        if not self.transformers:
            self.load_transformer_data()

        if not self.load_factors:
            self.analyze_transformer_capacity()

        if not self.network:
            self.build_network_model()

        if not self.substation_clusters:
            self.identify_transformer_clusters()

        if not self.reinforcement_priorities:
            self.prioritize_reinforcements()

        # Create summary statistics
        summary = {
            'total_transformers': len(self.transformers),
            'cities': list(set(transformer.get('city', 'Unknown') for transformer in self.transformers.values())),
            'avg_load_factor': np.mean(list(self.load_factors.values())),
            'clusters': len(set(self.substation_clusters.values())),
            'high_priority_transformers': [tid for tid, score in self.reinforcement_priorities[:5]]
        }

        # Generate city-level statistics
        city_stats = {}
        for transformer_id, transformer in self.transformers.items():
            city = transformer.get('city', 'Unknown')

            if city not in city_stats:
                city_stats[city] = {
                    'count': 0,
                    'avg_load_factor': 0,
                    'total_load': 0,
                    'transformers': []
                }

            city_stats[city]['count'] += 1
            city_stats[city]['transformers'].append(transformer_id)

            if 'metrics' in transformer:
                city_stats[city]['avg_load_factor'] += transformer['metrics']['load_factor']
                city_stats[city]['total_load'] += transformer['metrics']['max_load']

        # Calculate averages
        for city in city_stats:
            if city_stats[city]['count'] > 0:
                city_stats[city]['avg_load_factor'] /= city_stats[city]['count']

        # Create summary visualization
        plt.figure(figsize=(14, 10))

        # Plot 1: Transformer count by city
        plt.subplot(2, 2, 1)
        cities = list(city_stats.keys())
        counts = [city_stats[city]['count'] for city in cities]

        plt.bar(cities, counts, color='lightblue')
        plt.title('Transformers by City')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')

        # Plot 2: Average load factor by city
        plt.subplot(2, 2, 2)
        load_factors = [city_stats[city]['avg_load_factor'] for city in cities]

        plt.bar(cities, load_factors, color='salmon')
        plt.title('Average Load Factor by City')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Load Factor')

        # Plot 3: Total load by city
        plt.subplot(2, 2, 3)
        total_loads = [city_stats[city]['total_load'] for city in cities]

        plt.bar(cities, total_loads, color='lightgreen')
        plt.title('Total Load by City')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Total Load')

        # Plot 4: Priority score distribution
        plt.subplot(2, 2, 4)
        priority_scores = [score for _, score in self.reinforcement_priorities]

        plt.hist(priority_scores, bins=10, color='purple', alpha=0.7)
        plt.title('Distribution of Priority Scores')
        plt.xlabel('Priority Score')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig('brussels_summary_report.png')
        plt.close()

        # Create a complete report dictionary
        report = {
            'summary': summary,
            'city_stats': city_stats,
            'top_priorities': self.reinforcement_priorities[:10],
            'visualizations': [
                'brussels_summary_report.png',
                'brussels_substation_network.png',
                'brussels_transformer_clusters.png',
                'brussels_reinforcement_priorities.png',
                'brussels_city_priority_heatmap.png'
            ],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save report to JSON
        with open('brussels_substation_report.json', 'w') as f:
            # Convert data to serializable format
            serializable_report = {
                'summary': summary,
                'city_stats': {
                    city: {
                        'count': stats['count'],
                        'avg_load_factor': stats['avg_load_factor'],
                        'total_load': stats['total_load'],
                        'transformers': stats['transformers']
                    }
                    for city, stats in city_stats.items()
                },
                'top_priorities': [
                    {'transformer_id': tid, 'score': float(score)}
                    for tid, score in self.reinforcement_priorities[:10]
                ],
                'visualizations': report['visualizations'],
                'timestamp': report['timestamp']
            }

            json.dump(serializable_report, f, indent=4)

        return report


# Create an instance of the SubstationManager class
manager = SubstationManager(
    models_dir='saved_models',  # Directory containing saved transformer models
    results_dir='results',  # Directory containing transformer data
    prediction_dir='test'  # Directory containing prediction results
)

if __name__ == '__main__':
    # Instantiate the manager
    manager = SubstationManager()

    # Load transformer data (this will print a message)
    manager.load_transformer_data()

    # Analyze transformer capacity
    manager.analyze_transformer_capacity()

    # Build the network model
    manager.build_network_model()

    # Identify clusters and print the result
    clusters = manager.identify_transformer_clusters()
    print("Transformer clusters:", clusters)

    # Visualize the network and print the output file name
    network_file = manager.visualize_network_by_city()
    print("Network visualization saved as:", network_file)

    # Prioritize reinforcements and print the result
    priorities = manager.prioritize_reinforcements()
    print("Reinforcement priorities:", priorities)

    # Generate a summary report and print the summary
    summary = manager.generate_summary_report()
    print("Summary report generated at:", summary.get('timestamp'))
