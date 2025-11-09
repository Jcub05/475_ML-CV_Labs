"""
Generate comparison table for Lab 3 report
Aggregates results from all trained models
"""

import pickle
from pathlib import Path
from tabulate import tabulate
import numpy as np

def load_results(results_dir='./results'):
    """Load all results pickle files"""
    results_dir = Path(results_dir)
    results = []
    
    # Model and KD mode combinations
    models = ['ultracompact', 'standard']
    kd_modes = ['none', 'response', 'feature']
    
    for model in models:
        for kd_mode in kd_modes:
            pkl_path = results_dir / f'results_{model}_{kd_mode}.pkl'
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    result = pickle.load(f)
                    results.append(result)
            else:
                print(f"Warning: {pkl_path} not found")
    
    return results


def load_training_info(checkpoints_dir='./checkpoints'):
    """Load training time information from checkpoint files"""
    import torch
    
    checkpoints_dir = Path(checkpoints_dir)
    training_info = {}
    
    models = ['ultracompact', 'standard']
    kd_modes = ['none', 'response', 'feature']
    
    for model in models:
        for kd_mode in kd_modes:
            ckpt_path = checkpoints_dir / f'best_model_{model}_{kd_mode}.pth'
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'history' in checkpoint:
                    history = checkpoint['history']
                    if 'epoch_times' in history:
                        total_time = sum(history['epoch_times'])
                        training_info[f'{model}_{kd_mode}'] = {
                            'total_time': total_time,
                            'epochs': len(history['epoch_times']),
                            'avg_epoch_time': np.mean(history['epoch_times'])
                        }
    
    return training_info


def generate_table(results, training_info=None):
    """Generate comparison table"""
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'KD Method', 'Parameters', 'mIoU', 'Inference (ms)', 'FPS', 'Score', 'Training Time']
    
    for result in results:
        model_name = result['model_type'].capitalize()
        kd_name = result['kd_mode'].capitalize()
        if kd_name == 'None':
            kd_name = 'Without KD'
        elif kd_name == 'Response':
            kd_name = 'Response-based'
        elif kd_name == 'Feature':
            kd_name = 'Feature-based'
        
        params = f"{result['num_params'] / 1e6:.2f}M"
        miou = f"{result['mean_iou']:.4f}"
        inference_time = f"{result['avg_inference_time']:.2f}"
        fps = f"{result['fps']:.1f}" if 'fps' in result else "N/A"
        score = f"{result['score']:.4f}"
        
        # Get training time if available
        key = f"{result['model_type']}_{result['kd_mode']}"
        if training_info and key in training_info:
            train_time = f"{training_info[key]['total_time']/3600:.2f}h"
        else:
            train_time = "N/A"
        
        table_data.append([model_name, kd_name, params, miou, inference_time, fps, score, train_time])
    
    # Sort by model first, then by KD method
    kd_order = {'Without KD': 0, 'Response-based': 1, 'Feature-based': 2}
    table_data.sort(key=lambda x: (x[0], kd_order.get(x[1], 3)))
    
    return headers, table_data


def print_detailed_comparison(results, training_info=None):
    """Print detailed comparison including per-class IoU"""
    
    print("\n" + "=" * 100)
    print("DETAILED COMPARISON ACROSS ALL MODELS")
    print("=" * 100)
    
    # Main comparison table
    headers, table_data = generate_table(results, training_info)
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Find best models
    print("\n" + "=" * 100)
    print("BEST MODELS BY METRIC:")
    print("=" * 100)
    
    best_miou = max(results, key=lambda x: x['mean_iou'])
    best_score = max(results, key=lambda x: x['score'])
    best_inference = min(results, key=lambda x: x['avg_inference_time'])
    
    print(f"\nBest mIoU: {best_miou['model_type']}-{best_miou['kd_mode']} ({best_miou['mean_iou']:.4f})")
    print(f"Best Score: {best_score['model_type']}-{best_score['kd_mode']} ({best_score['score']:.4f})")
    print(f"Fastest Inference: {best_inference['model_type']}-{best_inference['kd_mode']} ({best_inference['avg_inference_time']:.2f} ms)")
    
    # Analysis
    print("\n" + "=" * 100)
    print("KNOWLEDGE DISTILLATION IMPACT:")
    print("=" * 100)
    
    for model_type in ['ultracompact', 'standard']:
        print(f"\n{model_type.upper()} Model:")
        model_results = [r for r in results if r['model_type'] == model_type]
        
        if len(model_results) >= 2:
            baseline = next((r for r in model_results if r['kd_mode'] == 'none'), None)
            if baseline:
                for result in model_results:
                    if result['kd_mode'] != 'none':
                        miou_improvement = (result['mean_iou'] - baseline['mean_iou']) / baseline['mean_iou'] * 100
                        score_improvement = (result['score'] - baseline['score']) / baseline['score'] * 100
                        print(f"  {result['kd_mode'].capitalize()} KD:")
                        print(f"    mIoU improvement: {miou_improvement:+.2f}%")
                        print(f"    Score improvement: {score_improvement:+.2f}%")
    
    # Per-class IoU comparison (optional, can be very long)
    print("\n" + "=" * 100)
    print("PER-CLASS IOU SUMMARY (Average across all models):")
    print("=" * 100)
    
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
        'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
        'train', 'tv/monitor'
    ]
    
    # Average per-class IoU across all models
    avg_class_ious = {}
    for cls_idx, cls_name in enumerate(voc_classes):
        ious = [r['class_ious'].get(cls_idx, 0) for r in results if cls_idx in r['class_ious']]
        if ious:
            avg_class_ious[cls_name] = np.mean(ious)
    
    # Sort by IoU
    sorted_classes = sorted(avg_class_ious.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 best-segmented classes:")
    for i, (cls_name, iou) in enumerate(sorted_classes[:5], 1):
        print(f"  {i}. {cls_name}: {iou:.4f}")
    
    print("\nBottom 5 worst-segmented classes:")
    for i, (cls_name, iou) in enumerate(sorted_classes[-5:], 1):
        print(f"  {i}. {cls_name}: {iou:.4f}")


def save_latex_table(results, training_info=None, output_path='comparison_table.tex'):
    """Save comparison table in LaTeX format for report"""
    
    headers, table_data = generate_table(results, training_info)
    
    with open(output_path, 'w') as f:
        f.write("% Comparison table for Lab 3 report\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of Knowledge Distillation Methods}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{|l|l|r|r|r|r|r|r|}\n")
        f.write("\\hline\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\hline\n")
        
        for row in table_data:
            f.write(" & ".join(str(x) for x in row) + " \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nLaTeX table saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comparison table from test results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing results pickle files')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoint files')
    parser.add_argument('--save_latex', action='store_true',
                        help='Save table in LaTeX format')
    parser.add_argument('--output', type=str, default='comparison_table.tex',
                        help='Output file for LaTeX table')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("Error: No results found!")
        print(f"Make sure you have run test.py for all model variants and results are in {args.results_dir}")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Load training info
    print("Loading training information...")
    training_info = load_training_info(args.checkpoints_dir)
    
    # Print detailed comparison
    print_detailed_comparison(results, training_info)
    
    # Save LaTeX table if requested
    if args.save_latex:
        save_latex_table(results, training_info, args.output)
    
    # Save text version
    text_output = Path(args.results_dir) / 'comparison_table.txt'
    headers, table_data = generate_table(results, training_info)
    with open(text_output, 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nText table saved to: {text_output}")
    
    print("\n" + "=" * 100)
    print("Comparison table generation complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
