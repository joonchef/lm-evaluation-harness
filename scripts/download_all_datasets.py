#!/usr/bin/env python3
"""
모든 lm-evaluation-harness task의 데이터셋을 다운로드하고 캐시하는 스크립트

Usage:
    python download_all_datasets.py --cache-dir /path/to/cache --tasks all
    python download_all_datasets.py --cache-dir /path/to/cache --tasks hellaswag,arc_easy
    python download_all_datasets.py --cache-dir /path/to/cache --task-file tasks.txt
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml

try:
    import datasets
    from datasets import load_dataset, DownloadMode
    from tqdm import tqdm
except ImportError:
    print("Required packages not found. Please install:")
    print("pip install datasets tqdm pyyaml")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """lm-evaluation-harness 데이터셋 다운로더"""
    
    def __init__(self, cache_dir: str, lm_eval_path: Optional[str] = None):
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # lm-evaluation-harness 경로 찾기
        if lm_eval_path:
            self.lm_eval_path = Path(lm_eval_path)
        else:
            # 스크립트 위치에서 프로젝트 루트 찾기
            self.lm_eval_path = Path(__file__).parent.parent
        
        self.tasks_dir = self.lm_eval_path / "lm_eval" / "tasks"
        
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"Tasks directory not found: {self.tasks_dir}")
        
        self.downloaded_datasets = set()
        self.failed_datasets = []
        
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Tasks directory: {self.tasks_dir}")
    
    def parse_yaml_file(self, yaml_path: Path) -> Optional[Dict]:
        """YAML 파일 파싱"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content
        except Exception as e:
            logger.warning(f"Failed to parse {yaml_path}: {e}")
            return None
    
    def extract_dataset_info(self, config: Dict) -> Optional[tuple]:
        """YAML 설정에서 데이터셋 정보 추출"""
        if not config:
            return None
        
        # dataset_path와 dataset_name 추출
        dataset_path = config.get('dataset_path')
        dataset_name = config.get('dataset_name')
        
        if dataset_path:
            # 로컬 경로가 아닌 HuggingFace 데이터셋인 경우만
            if not dataset_path.startswith('/') and not dataset_path.startswith('.'):
                return (dataset_path, dataset_name)
        
        return None
    
    def find_all_tasks(self) -> Dict[str, Dict]:
        """모든 task 찾기"""
        tasks = {}
        
        # 모든 YAML 파일 검색
        for yaml_file in self.tasks_dir.rglob("*.yaml"):
            config = self.parse_yaml_file(yaml_file)
            if config:
                task_name = config.get('task')
                if task_name:
                    dataset_info = self.extract_dataset_info(config)
                    if dataset_info:
                        tasks[task_name] = {
                            'yaml_path': str(yaml_file),
                            'dataset_path': dataset_info[0],
                            'dataset_name': dataset_info[1],
                        }
        
        logger.info(f"Found {len(tasks)} tasks with datasets")
        return tasks
    
    def download_dataset(self, dataset_path: str, dataset_name: Optional[str] = None) -> bool:
        """단일 데이터셋 다운로드"""
        dataset_key = f"{dataset_path}/{dataset_name}" if dataset_name else dataset_path
        
        # 이미 다운로드한 경우 건너뛰기
        if dataset_key in self.downloaded_datasets:
            logger.debug(f"Already downloaded: {dataset_key}")
            return True
        
        try:
            logger.info(f"Downloading: {dataset_key}")
            
            # 데이터셋 다운로드
            dataset = load_dataset(
                path=dataset_path,
                name=dataset_name,
                cache_dir=str(self.cache_dir),
                download_mode=DownloadMode.FORCE_REDOWNLOAD,  # 강제 다운로드
                trust_remote_code=True,  # 일부 데이터셋에 필요
            )
            
            # 성공 기록
            self.downloaded_datasets.add(dataset_key)
            logger.info(f"✓ Successfully downloaded: {dataset_key}")
            
            # 데이터셋 정보 저장
            info = {
                'dataset_path': dataset_path,
                'dataset_name': dataset_name,
                'splits': list(dataset.keys()) if dataset else [],
                'cache_dir': str(self.cache_dir),
            }
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to download {dataset_key}: {str(e)}"
            logger.error(error_msg)
            self.failed_datasets.append({
                'dataset': dataset_key,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def download_tasks(self, task_names: Optional[List[str]] = None):
        """지정된 task들의 데이터셋 다운로드"""
        # 모든 task 정보 수집
        all_tasks = self.find_all_tasks()
        
        # 다운로드할 task 결정
        if task_names and task_names != ['all']:
            tasks_to_download = {
                name: info for name, info in all_tasks.items() 
                if name in task_names
            }
            
            # 찾지 못한 task 경고
            not_found = set(task_names) - set(tasks_to_download.keys())
            if not_found:
                logger.warning(f"Tasks not found: {not_found}")
        else:
            tasks_to_download = all_tasks
        
        logger.info(f"Will download datasets for {len(tasks_to_download)} tasks")
        
        # 중복 제거를 위한 unique datasets
        unique_datasets = {}
        for task_name, info in tasks_to_download.items():
            dataset_key = f"{info['dataset_path']}/{info['dataset_name']}" if info['dataset_name'] else info['dataset_path']
            if dataset_key not in unique_datasets:
                unique_datasets[dataset_key] = info
        
        logger.info(f"Unique datasets to download: {len(unique_datasets)}")
        
        # 다운로드 실행
        with tqdm(total=len(unique_datasets), desc="Downloading datasets") as pbar:
            for dataset_key, info in unique_datasets.items():
                success = self.download_dataset(
                    info['dataset_path'], 
                    info['dataset_name']
                )
                pbar.update(1)
                pbar.set_postfix({'current': dataset_key, 'success': success})
    
    def save_summary(self):
        """다운로드 요약 저장"""
        summary_path = self.cache_dir / "download_summary.json"
        
        summary = {
            'cache_dir': str(self.cache_dir),
            'total_downloaded': len(self.downloaded_datasets),
            'downloaded_datasets': sorted(list(self.downloaded_datasets)),
            'failed_count': len(self.failed_datasets),
            'failed_datasets': self.failed_datasets,
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to: {summary_path}")
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        print(f"Successfully downloaded: {len(self.downloaded_datasets)} datasets")
        print(f"Failed: {len(self.failed_datasets)} datasets")
        print(f"Cache directory: {self.cache_dir}")
        
        if self.failed_datasets:
            print("\nFailed datasets:")
            for item in self.failed_datasets:
                print(f"  - {item['dataset']}: {item['error']}")
        
        print("\nTo use this cache in offline mode:")
        print(f"  export HF_DATASETS_CACHE={self.cache_dir}")
        print(f"  export HF_DATASETS_OFFLINE=1")


def main():
    parser = argparse.ArgumentParser(
        description="Download all datasets for lm-evaluation-harness tasks"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory to store downloaded datasets"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks or 'all' for all tasks"
    )
    parser.add_argument(
        "--task-file",
        type=str,
        help="File containing list of tasks (one per line)"
    )
    parser.add_argument(
        "--lm-eval-path",
        type=str,
        help="Path to lm-evaluation-harness repository"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Task 목록 결정
    task_names = None
    if args.tasks:
        if args.tasks.lower() == 'all':
            task_names = ['all']
        else:
            task_names = [t.strip() for t in args.tasks.split(',')]
    elif args.task_file:
        with open(args.task_file, 'r') as f:
            task_names = [line.strip() for line in f if line.strip()]
    
    # 다운로더 실행
    downloader = DatasetDownloader(args.cache_dir, args.lm_eval_path)
    
    try:
        downloader.download_tasks(task_names)
        downloader.save_summary()
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        downloader.save_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()