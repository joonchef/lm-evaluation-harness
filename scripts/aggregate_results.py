#!/usr/bin/env python3
"""
여러 lm-evaluation-harness 실행 결과를 통합하고 정리하는 스크립트

Usage:
    python aggregate_results.py --results-dir results/ --output comparison.md
    python aggregate_results.py --results-dir results/ --format csv --output comparison.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultsAggregator:
    """평가 결과를 수집하고 정리하는 클래스"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.experiments = defaultdict(dict)
        
    def load_results(self, pattern: str = "results*.json") -> None:
        """지정된 디렉토리에서 모든 결과 JSON 파일 로드"""
        json_files = list(self.results_dir.rglob(pattern))
        logger.info(f"Found {len(json_files)} result files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 실험 메타데이터 추출
                metadata = self._extract_metadata(data, json_file)
                
                # 결과 저장
                self.all_results.append({
                    'file': str(json_file),
                    'metadata': metadata,
                    'results': data.get('results', {}),
                    'configs': data.get('configs', {}),
                })
                
                logger.info(f"Loaded: {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    def _extract_metadata(self, data: Dict, filepath: Path) -> Dict:
        """JSON 데이터와 파일 경로에서 메타데이터 추출"""
        metadata = {
            'model': data.get('model_name', 'unknown'),
            'model_args': data.get('model_args', ''),
            'num_fewshot': data.get('num_fewshot', 'N/A'),
            'batch_size': data.get('batch_size', 'N/A'),
            'device': data.get('device', 'N/A'),
            'limit': data.get('limit', None),
            'timestamp': data.get('date', 'N/A'),
        }
        
        # 파일 경로에서 추가 정보 추출 (선택적)
        path_parts = filepath.parts
        if len(path_parts) > 2:
            metadata['experiment_group'] = path_parts[-2]
        
        return metadata
    
    def create_comparison_table(self) -> pd.DataFrame:
        """모든 결과를 비교 테이블로 변환"""
        rows = []
        
        for result_data in self.all_results:
            metadata = result_data['metadata']
            results = result_data['results']
            
            for task_name, task_results in results.items():
                row = {
                    'Model': metadata['model'],
                    'Task': task_name,
                    'Few-shot': metadata['num_fewshot'],
                    'Batch Size': metadata['batch_size'],
                }
                
                # 주요 메트릭 추출
                if isinstance(task_results, dict):
                    # 정확도 관련 메트릭
                    for metric in ['acc', 'acc_norm', 'em', 'f1', 'bleu', 'rouge1', 'rouge2', 'rougeL']:
                        metric_key = f"{metric},none"
                        if metric_key in task_results:
                            row[metric.upper()] = task_results[metric_key]
                            stderr_key = f"{metric}_stderr,none"
                            if stderr_key in task_results:
                                row[f"{metric.upper()}_stderr"] = task_results[stderr_key]
                    
                    # Perplexity
                    if 'perplexity' in task_results:
                        row['Perplexity'] = task_results['perplexity']
                    
                    # Alias 있는 경우
                    if 'alias' in task_results:
                        row['Task'] = task_results.get('alias', task_name)
                
                # 메타데이터 추가
                row['Limit'] = metadata['limit']
                row['Timestamp'] = metadata['timestamp']
                row['File'] = result_data['file']
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 정렬
        if not df.empty:
            sort_columns = ['Model', 'Task', 'Few-shot']
            existing_sort_cols = [col for col in sort_columns if col in df.columns]
            df = df.sort_values(by=existing_sort_cols)
        
        return df
    
    def create_pivot_table(self, metric: str = 'ACC') -> pd.DataFrame:
        """피벗 테이블 생성 (Task x Model/Settings)"""
        df = self.create_comparison_table()
        
        if df.empty or metric not in df.columns:
            logger.warning(f"No data available for metric: {metric}")
            return pd.DataFrame()
        
        # 피벗 테이블 생성
        pivot = df.pivot_table(
            index='Task',
            columns=['Model', 'Few-shot'],
            values=metric,
            aggfunc='first'  # 중복된 경우 첫 번째 값 사용
        )
        
        # 소수점 정리
        pivot = pivot.round(4)
        
        return pivot
    
    def create_summary_stats(self) -> pd.DataFrame:
        """요약 통계 생성"""
        df = self.create_comparison_table()
        
        if df.empty:
            return pd.DataFrame()
        
        # 모델별 평균 성능
        summary = []
        
        for model in df['Model'].unique():
            model_df = df[df['Model'] == model]
            
            stats = {
                'Model': model,
                'Num Tasks': model_df['Task'].nunique(),
                'Num Configurations': len(model_df),
            }
            
            # 각 메트릭의 평균 계산
            for metric in ['ACC', 'ACC_NORM', 'F1', 'EM']:
                if metric in model_df.columns:
                    values = model_df[metric].dropna()
                    if not values.empty:
                        stats[f'Avg {metric}'] = values.mean()
                        stats[f'Std {metric}'] = values.std()
                        stats[f'Min {metric}'] = values.min()
                        stats[f'Max {metric}'] = values.max()
            
            summary.append(stats)
        
        return pd.DataFrame(summary)
    
    def export_markdown(self, output_file: str) -> None:
        """결과를 Markdown 파일로 내보내기"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# LM Evaluation Results Summary\n\n")
            
            # 요약 통계
            f.write("## Summary Statistics\n\n")
            summary = self.create_summary_stats()
            if not summary.empty:
                f.write(summary.to_markdown(index=False, floatfmt=".4f"))
            else:
                f.write("No summary statistics available.\n")
            f.write("\n\n")
            
            # 전체 결과 테이블
            f.write("## Detailed Results\n\n")
            full_table = self.create_comparison_table()
            if not full_table.empty:
                # 표시할 컬럼 선택
                display_cols = ['Model', 'Task', 'Few-shot']
                metric_cols = [col for col in full_table.columns 
                             if col.upper() in ['ACC', 'ACC_NORM', 'F1', 'EM', 'PERPLEXITY']]
                display_cols.extend(metric_cols)
                
                display_df = full_table[display_cols].dropna(axis=1, how='all')
                f.write(display_df.to_markdown(index=False, floatfmt=".4f"))
            else:
                f.write("No detailed results available.\n")
            f.write("\n\n")
            
            # 피벗 테이블 (ACC 기준)
            f.write("## Performance Matrix (Accuracy)\n\n")
            pivot = self.create_pivot_table('ACC')
            if not pivot.empty:
                f.write(pivot.to_markdown(floatfmt=".4f"))
            else:
                f.write("No accuracy data available for pivot table.\n")
            f.write("\n\n")
            
            # 파일 목록
            f.write("## Source Files\n\n")
            for result in self.all_results:
                f.write(f"- {result['file']}\n")
        
        logger.info(f"Markdown report saved to: {output_file}")
    
    def export_csv(self, output_file: str) -> None:
        """결과를 CSV 파일로 내보내기"""
        df = self.create_comparison_table()
        
        if not df.empty:
            # 불필요한 컬럼 제거
            export_cols = [col for col in df.columns if col not in ['File', 'Timestamp']]
            df[export_cols].to_csv(output_file, index=False)
            logger.info(f"CSV saved to: {output_file}")
        else:
            logger.warning("No data to export to CSV")
    
    def export_latex(self, output_file: str) -> None:
        """결과를 LaTeX 테이블로 내보내기"""
        pivot = self.create_pivot_table('ACC')
        
        if not pivot.empty:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("% LM Evaluation Results Table\n")
                f.write(pivot.to_latex(float_format="%.4f"))
            logger.info(f"LaTeX table saved to: {output_file}")
        else:
            logger.warning("No data to export to LaTeX")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and compare lm-evaluation-harness results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison.md",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "csv", "latex"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results*.json",
        help="Pattern for matching result files"
    )
    
    args = parser.parse_args()
    
    # 결과 수집 및 처리
    aggregator = ResultsAggregator(args.results_dir)
    aggregator.load_results(args.pattern)
    
    # 결과가 없는 경우
    if not aggregator.all_results:
        logger.error("No results found!")
        return
    
    # 형식에 따라 내보내기
    if args.format == "markdown":
        aggregator.export_markdown(args.output)
    elif args.format == "csv":
        aggregator.export_csv(args.output)
    elif args.format == "latex":
        aggregator.export_latex(args.output)
    
    logger.info(f"Results aggregation complete: {args.output}")


if __name__ == "__main__":
    main()