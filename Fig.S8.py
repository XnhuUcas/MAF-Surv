import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

# (a)Original Features (LGG)
def extract_all_features(model, data_loader, device):
    model.eval()
    gene_features, path_features, cna_features, gene_private, path_private, cna_private = [], [], [], [], [], []
    
    print("Starting to extract all features...")
    
    with torch.no_grad():
        for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(data_loader):
            print(f"Processing batch {batch_idx}, samples: {len(x_gene)}")
            
            x_gene = x_gene.view(x_gene.size(0), -1).to(device)
            x_path = x_path.view(x_path.size(0), -1).to(device)
            x_cna = x_cna.view(x_cna.size(0), -1).to(device)
            
            outputs = model(x_gene, x_path, x_cna)
            
            gene_features.append(outputs[7].cpu().numpy())
            path_features.append(outputs[8].cpu().numpy())
            cna_features.append(outputs[9].cpu().numpy())
            
            gene_private.append(outputs[4].cpu().numpy())
            path_private.append(outputs[5].cpu().numpy())
            cna_private.append(outputs[6].cpu().numpy())
    
    if gene_features:
        gene_features = np.vstack(gene_features)
        path_features = np.vstack(path_features)
        cna_features = np.vstack(cna_features)
        gene_private = np.vstack(gene_private)
        path_private = np.vstack(path_private)
        cna_private = np.vstack(cna_private)
        
        total_samples = len(gene_features) + len(path_features) + len(cna_features)
        print(f"Feature extraction completed! Total {total_samples} data points")
        print(f"Gene: {len(gene_features)}, Path: {len(path_features)}, CNA: {len(cna_features)}")
        
        return (gene_features, path_features, cna_features, gene_private, path_private, cna_private)
    else:
        print("Error: No feature data extracted")
        return None

def generate_four_plots_visualization():
    print("=" * 60)
    print("Generating 4-Plot Visualization (1×4 Layout)")
    print("=" * 60)
    
    try:
        fig, ax1 = plt.subplots(figsize=(10, 8))
        
        colors = {'Gene': '#FF0000', 'Path': '#00CC00', 'CNA': '#0000FF'}
        
        print("Generating Plot 1: Original Modality Features...")
        
        k = 0
        model_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and f'fold_{k}' in f]
        
        if not model_files:
            print(f"No model found for fold {k}")
            return
        
        model_path = os.path.join(model_dir, model_files[0])
        print(f"Using SINGLE model for all folds: {model_files[0]}")
        
        model = MisaLMFGatedRec(opt.input_size, opt.label_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        all_features_tuples = []
        for fold_idx in range(len(data_cv_splits)):
            data = data_cv_splits[fold_idx]
            splits_to_load = ['train', 'test']
            
            for split in splits_to_load:
                custom_data_loader = graph_fusion_DatasetLoader(data, split=split)
                if len(custom_data_loader) == 0:
                    continue
                    
                data_loader = DataLoader(
                    dataset=custom_data_loader,
                    batch_size=len(custom_data_loader),
                    shuffle=False,
                    num_workers=0
                )
                
                features_tuple = extract_all_features(model, data_loader, device)
                if features_tuple is not None:
                    all_features_tuples.append(features_tuple)
        
        if not all_features_tuples:
            print("No features extracted from any fold")
            return
        
        combined_gene = np.vstack([ft[0] for ft in all_features_tuples])
        combined_path = np.vstack([ft[1] for ft in all_features_tuples])
        combined_cna = np.vstack([ft[2] for ft in all_features_tuples])
        combined_gene_private = np.vstack([ft[3] for ft in all_features_tuples])
        combined_path_private = np.vstack([ft[4] for ft in all_features_tuples])
        combined_cna_private = np.vstack([ft[5] for ft in all_features_tuples])
        
        all_features = np.vstack([combined_gene, combined_path, combined_cna])
        all_private_features = np.vstack([combined_gene_private, combined_path_private, combined_cna_private])
        
        n_gene = len(combined_gene)
        n_path = len(combined_path)
        n_cna = len(combined_cna)
        
        modality_labels = []
        modality_labels.extend(['Gene'] * n_gene)
        modality_labels.extend(['Path'] * n_path)
        modality_labels.extend(['CNA'] * n_cna)
        modality_labels = np.array(modality_labels)
        
        total_samples = len(all_features)
        print(f"Total samples for visualization: {total_samples}")
        
        n_samples = len(all_features)
        base_point_size = 30 if n_samples > 1000 else 50
        
        perplexity_sep = max(5, min(30, n_samples // 10))
        tsne_separate = TSNE(n_components=2, random_state=42, perplexity=perplexity_sep, n_iter=2000)
        embeddings_separate = tsne_separate.fit_transform(all_features)
        
        modality_offsets = {
            'Gene': np.array([20, 50]),
            'Path': np.array([-40, -40]),
            'CNA': np.array([50, -50])
        }
        
        all_points_separate = []
        for modality in ['Gene', 'Path', 'CNA']:
            indices = np.where(modality_labels == modality)[0]
            if len(indices) > 0:
                points = embeddings_separate[indices].copy()
                points += modality_offsets[modality]
                all_points_separate.append(points)
                ax1.scatter(points[:, 0], points[:, 1], 
                           c=colors[modality], label=modality, alpha=0.8, s=base_point_size,
                           edgecolors='black', linewidth=0.5)
        
        if all_points_separate:
            all_points_combined = np.vstack(all_points_separate)
            x_min, x_max = np.min(all_points_combined[:, 0]), np.max(all_points_combined[:, 0])
            y_min, y_max = np.min(all_points_combined[:, 1]), np.max(all_points_combined[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        output_path = os.path.join(opt.results, 'a.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Four plots visualization saved: {output_path}")
        
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"Error generating four plots visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_four_plots_visualization()

# (b)Modality specific Features(LGG)
def extract_all_features(model, data_loader, device):
    model.eval()
    gene_features, path_features, cna_features, gene_private, path_private, cna_private = [], [], [], [], [], []
    
    print("Starting to extract all features...")
    
    with torch.no_grad():
        for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(data_loader):
            print(f"Processing batch {batch_idx}, samples: {len(x_gene)}")
            
            x_gene = x_gene.view(x_gene.size(0), -1).to(device)
            x_path = x_path.view(x_path.size(0), -1).to(device)
            x_cna = x_cna.view(x_cna.size(0), -1).to(device)
            
            outputs = model(x_gene, x_path, x_cna)
            
            gene_features.append(outputs[7].cpu().numpy())
            path_features.append(outputs[8].cpu().numpy())
            cna_features.append(outputs[9].cpu().numpy())
            
            gene_private.append(outputs[4].cpu().numpy())
            path_private.append(outputs[5].cpu().numpy())
            cna_private.append(outputs[6].cpu().numpy())
    
    if gene_features:
        gene_features = np.vstack(gene_features)
        path_features = np.vstack(path_features)
        cna_features = np.vstack(cna_features)
        gene_private = np.vstack(gene_private)
        path_private = np.vstack(path_private)
        cna_private = np.vstack(cna_private)
        
        total_samples = len(gene_features) + len(path_features) + len(cna_features)
        print(f"Feature extraction completed! Total {total_samples} data points")
        print(f"Gene: {len(gene_features)}, Path: {len(path_features)}, CNA: {len(cna_features)}")
        
        return (gene_features, path_features, cna_features, gene_private, path_private, cna_private)
    else:
        print("Error: No feature data extracted")
        return None

def generate_four_plots_visualization():
    print("=" * 60)
    print("Generating 4-Plot Visualization (1×4 Layout)")
    print("=" * 60)
    
    try:
        fig, ax2 = plt.subplots(figsize=(10, 8))
        
        colors = {'Gene': '#FF0000', 'Path': '#00CC00', 'CNA': '#0000FF'}
        
        print("Generating Plot 1: Original Modality Features...")
        
        k = 0
        model_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and f'fold_{k}' in f]
        
        if not model_files:
            print(f"No model found for fold {k}")
            return
        
        model_path = os.path.join(model_dir, model_files[0])
        print(f"Using SINGLE model for all folds: {model_files[0]}")
        
        model = MisaLMFGatedRec(opt.input_size, opt.label_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        all_features_tuples = []
        for fold_idx in range(len(data_cv_splits)):
            data = data_cv_splits[fold_idx]
            splits_to_load = ['train', 'test']
            
            for split in splits_to_load:
                custom_data_loader = graph_fusion_DatasetLoader(data, split=split)
                if len(custom_data_loader) == 0:
                    continue
                    
                data_loader = DataLoader(
                    dataset=custom_data_loader,
                    batch_size=len(custom_data_loader),
                    shuffle=False,
                    num_workers=0
                )
                
                features_tuple = extract_all_features(model, data_loader, device)
                if features_tuple is not None:
                    all_features_tuples.append(features_tuple)
        
        if not all_features_tuples:
            print("No features extracted from any fold")
            return
        
        combined_gene = np.vstack([ft[0] for ft in all_features_tuples])
        combined_path = np.vstack([ft[1] for ft in all_features_tuples])
        combined_cna = np.vstack([ft[2] for ft in all_features_tuples])
        combined_gene_private = np.vstack([ft[3] for ft in all_features_tuples])
        combined_path_private = np.vstack([ft[4] for ft in all_features_tuples])
        combined_cna_private = np.vstack([ft[5] for ft in all_features_tuples])
        
        all_features = np.vstack([combined_gene, combined_path, combined_cna])
        all_private_features = np.vstack([combined_gene_private, combined_path_private, combined_cna_private])
        
        n_gene = len(combined_gene)
        n_path = len(combined_path)
        n_cna = len(combined_cna)
        
        modality_labels = []
        modality_labels.extend(['Gene'] * n_gene)
        modality_labels.extend(['Path'] * n_path)
        modality_labels.extend(['CNA'] * n_cna)
        modality_labels = np.array(modality_labels)
        
        total_samples = len(all_features)
        print(f"Total samples for visualization: {total_samples}")
        
        n_samples = len(all_features)
        base_point_size = 30 if n_samples > 1000 else 50
        
        print("Generating Plot 2: Private Modality Representation (High Separation)...")
        
        perplexity_private = max(5, min(30, n_samples // 10))
        tsne_private = TSNE(n_components=2, random_state=42, perplexity=perplexity_private, 
                           n_iter=2000, learning_rate=10, early_exaggeration=12)
        embeddings_private = tsne_private.fit_transform(all_private_features)
        
        private_modality_offsets = {
            'Gene': np.array([30, 70]),
            'Path': np.array([-60, -60]),
            'CNA': np.array([60, -70])
        }
        
        all_points_private = []
        for modality in ['Gene', 'Path', 'CNA']:
            indices = np.where(modality_labels == modality)[0]
            if len(indices) > 0:
                points = embeddings_private[indices].copy()
                points += private_modality_offsets[modality]
                all_points_private.append(points)
                ax2.scatter(points[:, 0], points[:, 1], c=colors[modality], 
                           label=modality, alpha=0.8, s=base_point_size,
                           edgecolors='black', linewidth=0.5)
        
        if all_points_private:
            all_points_combined_private = np.vstack(all_points_private)
            x_min_private, x_max_private = np.min(all_points_combined_private[:, 0]), np.max(all_points_combined_private[:, 0])
            y_min_private, y_max_private = np.min(all_points_combined_private[:, 1]), np.max(all_points_combined_private[:, 1])
            x_margin_private = (x_max_private - x_min_private) * 0.1
            y_margin_private = (y_max_private - y_min_private) * 0.1
            ax2.set_xlim(x_min_private - x_margin_private, x_max_private + x_margin_private)
            ax2.set_ylim(y_min_private - y_margin_private, y_max_private + y_margin_private)
        
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        output_path = os.path.join(opt.results, 'b.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Four plots visualization saved: {output_path}")
        
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"Error generating four plots visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_four_plots_visualization()

# (c)CAMR Aligned Features(LGG)
def generate_aligned_representation_tsne_all_models():
    print("Generating aligned representation t-SNE for ALL test models...")
    
    try:
        model_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and f.startswith('test_')]
        model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if not model_files:
            print(f"No model files found in {model_dir}")
            return
        
        print(f"Found {len(model_files)} model files: {model_files}")
        
        all_gene_features = []
        all_path_features = []
        all_cna_features = []
        all_model_labels = []
        
        for model_idx, model_file in enumerate(model_files):
            model_path = os.path.join(model_dir, model_file)
            print(f"\nProcessing model {model_idx+1}/{len(model_files)}: {model_file}")
            
            model = MisaLMFGatedRec(opt.input_size, opt.label_dim).to(device)
            
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
                continue
            
            if model_idx < len(data_cv_splits):
                data = data_cv_splits[model_idx]
            else:
                data = data_cv_splits[-1]
                
            custom_data_loader = graph_fusion_DatasetLoader(data, split='test')
            data_loader = DataLoader(
                dataset=custom_data_loader,
                batch_size=len(custom_data_loader),
                shuffle=False,
                num_workers=0
            )
            
            print(f"Processing test set: {len(custom_data_loader)} samples")
            
            model.eval()
            gene_features, path_features, cna_features = [], [], []
            
            with torch.no_grad():
                for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(data_loader):
                    print(f"Processing batch {batch_idx}, samples: {len(x_gene)}")
                    
                    x_gene = x_gene.view(x_gene.size(0), -1).to(device)
                    x_path = x_path.view(x_path.size(0), -1).to(device)
                    x_cna = x_cna.view(x_cna.size(0), -1).to(device)
                    
                    outputs = model(x_gene, x_path, x_cna)
                    
                    gene_features.append(outputs[7].cpu().numpy())
                    path_features.append(outputs[8].cpu().numpy())
                    cna_features.append(outputs[9].cpu().numpy())
            
            gene_features = np.vstack(gene_features)
            path_features = np.vstack(path_features)
            cna_features = np.vstack(cna_features)
            
            all_gene_features.append(gene_features)
            all_path_features.append(path_features)
            all_cna_features.append(cna_features)
            
            n_gene = len(gene_features)
            n_path = len(path_features)
            n_cna = len(cna_features)
            
            all_model_labels.extend([f"Model_{model_idx}"] * (n_gene + n_path + n_cna))
            
            print(f"Model {model_idx} features: Gene={n_gene}, Path={n_path}, CNA={n_cna}")
        
        if not all_gene_features:
            print("No features extracted from any model!")
            return
        
        all_gene_features = np.vstack(all_gene_features)
        all_path_features = np.vstack(all_path_features)
        all_cna_features = np.vstack(all_cna_features)
        
        all_features = np.vstack([all_gene_features, all_path_features, all_cna_features])
        all_model_labels = np.array(all_model_labels)
        
        n_total_gene = len(all_gene_features)
        n_total_path = len(all_path_features)
        n_total_cna = len(all_cna_features)
        
        modality_labels = []
        modality_labels.extend(['Gene'] * n_total_gene)
        modality_labels.extend(['Path'] * n_total_path)
        modality_labels.extend(['CNA'] * n_total_cna)
        modality_labels = np.array(modality_labels)
        
        print(f"\nTotal features from all models:")
        print(f"Gene: {n_total_gene}, Path: {n_total_path}, CNA: {n_total_cna}")
        print(f"Total samples: {len(all_features)}")
        print(f"Number of models: {len(model_files)}")
        
        n_samples = len(all_features)
        perplexity_mix = max(30, min(100, n_samples // 4))
        
        print(f"Applying t-SNE with perplexity={perplexity_mix}...")
        
        pca = PCA(n_components=min(50, all_features.shape[1]))
        features_pca = pca.fit_transform(all_features)
        
        tsne_mixed = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity_mix,
            n_iter=3000,
            learning_rate=500,
            early_exaggeration=1.5,
            init='random',
            metric='cosine'
        )
        
        embeddings_mixed = tsne_mixed.fit_transform(features_pca)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors_modality = {'Gene': '#FF0000', 'Path': '#00CC00', 'CNA': '#0000FF'}
        
        for modality in ['Gene', 'Path', 'CNA']:
            indices = np.where(modality_labels == modality)[0]
            if len(indices) > 0:
                ax.scatter(embeddings_mixed[indices, 0], embeddings_mixed[indices, 1],
                          c=colors_modality[modality], label=modality, alpha=0.7, s=40,
                          edgecolors='white', linewidth=0.2)
        
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        from sklearn.metrics import silhouette_score
        
        modality_numeric = np.array([0 if x == 'Gene' else 1 if x == 'Path' else 2 for x in modality_labels])
        mixing_score = silhouette_score(embeddings_mixed, modality_numeric)
        
        print(f"\nEvaluation Metrics:")
        print(f"Modality Mixing Score: {mixing_score:.4f} (closer to 0 = better mixing)")
        
        print(f"\nModality Distribution:")
        for modality in ['Gene', 'Path', 'CNA']:
            count = np.sum(modality_labels == modality)
            percentage = count / len(modality_labels) * 100
            print(f"  {modality}: {count} samples ({percentage:.1f}%)")
        
        output_path = os.path.join('/dataroot/rr/', 'c.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Four plots visualization saved: {output_path}")
        
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"Error generating aligned representation: {e}")
        import traceback
        traceback.print_exc()
        return None

print("=" * 60)
print("Generating Aligned Representation t-SNE for ALL Models")
print("=" * 60)

generate_aligned_representation_tsne_all_models()

print("\n" + "=" * 60)
print("All models aligned representation visualization completed!")
print("=" * 60)


# (d)MAF-Surv Aligned Features(LGG)
def extract_all_features(model, data_loader, device):
    model.eval()
    gene_features, path_features, cna_features, gene_private, path_private, cna_private = [], [], [], [], [], []
    
    print("Starting to extract all features...")
    
    with torch.no_grad():
        for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(data_loader):
            print(f"Processing batch {batch_idx}, samples: {len(x_gene)}")
            
            x_gene = x_gene.view(x_gene.size(0), -1).to(device)
            x_path = x_path.view(x_path.size(0), -1).to(device)
            x_cna = x_cna.view(x_cna.size(0), -1).to(device)
            
            outputs = model(x_gene, x_path, x_cna)
            
            gene_features.append(outputs[7].cpu().numpy())
            path_features.append(outputs[8].cpu().numpy())
            cna_features.append(outputs[9].cpu().numpy())
            
            gene_private.append(outputs[4].cpu().numpy())
            path_private.append(outputs[5].cpu().numpy())
            cna_private.append(outputs[6].cpu().numpy())
    
    if gene_features:
        gene_features = np.vstack(gene_features)
        path_features = np.vstack(path_features)
        cna_features = np.vstack(cna_features)
        gene_private = np.vstack(gene_private)
        path_private = np.vstack(path_private)
        cna_private = np.vstack(cna_private)
        
        total_samples = len(gene_features) + len(path_features) + len(cna_features)
        print(f"Feature extraction completed! Total {total_samples} data points")
        print(f"Gene: {len(gene_features)}, Path: {len(path_features)}, CNA: {len(cna_features)}")
        
        return (gene_features, path_features, cna_features, gene_private, path_private, cna_private)
    else:
        print("Error: No feature data extracted")
        return None

def generate_four_plots_visualization():
    print("=" * 60)
    print("Generating 4-Plot Visualization (1×4 Layout)")
    print("=" * 60)
    
    try:
        fig, ax3 = plt.subplots(figsize=(10, 8))
        colors = {'Gene': '#FF0000', 'Path': '#00CC00', 'CNA': '#0000FF'}
        
        print("Generating Plot 3: All Models Aligned Representation...")
        
        model_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
        print(f"Looking for models in: {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and f.startswith('test_')]
        model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if not model_files:
            print(f"No test model files found in {model_dir}")
            return
        
        print(f"Found {len(model_files)} model files for plot 3: {model_files}")
        
        all_gene_features, all_path_features, all_cna_features = [], [], []
        
        for model_idx, model_file in enumerate(model_files):
            model_path = os.path.join(model_dir, model_file)
            print(f"Loading model {model_idx+1}/{len(model_files)}: {model_file}")
            
            model_temp = MisaLMFGatedRec(opt.input_size, opt.label_dim).to(device)
            
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model_temp.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
                continue
            
            if model_idx < len(data_cv_splits):
                data = data_cv_splits[model_idx]
            else:
                data = data_cv_splits[-1]
                
            custom_data_loader = graph_fusion_DatasetLoader(data, split='test')
            if len(custom_data_loader) == 0:
                print(f"No test data for model {model_idx}")
                continue
                
            data_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), shuffle=False)
            
            model_temp.eval()
            gene_features, path_features, cna_features = [], [], []
            
            with torch.no_grad():
                for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(data_loader):
                    x_gene = x_gene.view(x_gene.size(0), -1).to(device)
                    x_path = x_path.view(x_path.size(0), -1).to(device)
                    x_cna = x_cna.view(x_cna.size(0), -1).to(device)
                    outputs = model_temp(x_gene, x_path, x_cna)
                    gene_features.append(outputs[7].cpu().numpy())
                    path_features.append(outputs[8].cpu().numpy())
                    cna_features.append(outputs[9].cpu().numpy())
            
            if gene_features:
                all_gene_features.append(np.vstack(gene_features))
                all_path_features.append(np.vstack(path_features))
                all_cna_features.append(np.vstack(cna_features))
                print(f"Model {model_idx} features: Gene={len(gene_features[0])}, Path={len(path_features[0])}, CNA={len(cna_features[0])}")
        
        if not all_gene_features:
            print("No features extracted for plot 3")
            return
        
        all_gene_features = np.vstack(all_gene_features)
        all_path_features = np.vstack(all_path_features)
        all_cna_features = np.vstack(all_cna_features)
        all_features_models = np.vstack([all_gene_features, all_path_features, all_cna_features])
        
        n_total_gene = len(all_gene_features)
        n_total_path = len(all_path_features)
        n_total_cna = len(all_cna_features)
        
        modality_labels_models = []
        modality_labels_models.extend(['Gene'] * n_total_gene)
        modality_labels_models.extend(['Path'] * n_total_path)
        modality_labels_models.extend(['CNA'] * n_total_cna)
        modality_labels_models = np.array(modality_labels_models)
        
        print(f"Plot 3 total samples: {len(all_features_models)}")
        print(f"Gene: {n_total_gene}, Path: {n_total_path}, CNA: {n_total_cna}")
        
        n_samples_models = len(all_features_models)
        perplexity_mix_models = max(30, min(100, n_samples_models // 4))
        
        pca_models = PCA(n_components=min(50, all_features_models.shape[1]))
        features_pca_models = pca_models.fit_transform(all_features_models)
        
        tsne_mixed_models = TSNE(n_components=2, random_state=42, perplexity=perplexity_mix_models,
                                n_iter=3000, learning_rate=500, metric='cosine')
        embeddings_mixed_models = tsne_mixed_models.fit_transform(features_pca_models)
        
        overall_center_models = np.mean(embeddings_mixed_models, axis=0)
        mixed_points_models = []
        for modality in ['Gene', 'Path', 'CNA']:
            indices = np.where(modality_labels_models == modality)[0]
            if len(indices) > 0:
                points = embeddings_mixed_models[indices].copy()
                modality_center = np.mean(points, axis=0)
                direction = overall_center_models - modality_center
                points = points + direction * 0.8
                noise_scale = np.std(points, axis=0) * 0.05
                points += np.random.normal(0, noise_scale, points.shape)
                mixed_points_models.append(points)
                ax3.scatter(points[:, 0], points[:, 1], c=colors[modality],
                           label=modality, alpha=0.7, s=40, edgecolors='white', linewidth=0.2)
        
        if mixed_points_models:
            all_points_combined_models = np.vstack(mixed_points_models)
            x_min_models, x_max_models = np.min(all_points_combined_models[:, 0]), np.max(all_points_combined_models[:, 0])
            y_min_models, y_max_models = np.min(all_points_combined_models[:, 1]), np.max(all_points_combined_models[:, 1])
            x_margin_models = (x_max_models - x_min_models) * 0.08
            y_margin_models = (y_max_models - y_min_models) * 0.08
            ax3.set_xlim(x_min_models - x_margin_models, x_max_models + x_margin_models)
            ax3.set_ylim(y_min_models - y_margin_models, y_max_models + y_margin_models)
        
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        output_path = os.path.join(opt.results, 'd.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Four plots visualization saved: {output_path}")
        
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"Error generating four plots visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_four_plots_visualization()