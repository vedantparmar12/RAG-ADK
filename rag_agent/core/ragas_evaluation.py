"""
RAGAS Evaluation Integration
Automated quality metrics for RAG systems using RAGAS framework
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

import numpy as np
import pandas as pd

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        answer_correctness, 
        faithfulness,
        context_precision,
        context_recall,
        context_relevancy,
        answer_similarity,
        context_entity_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    logging.warning("RAGAS not available. Install with: pip install ragas")
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RAGASMetrics:
    """Container for RAGAS evaluation metrics"""
    answer_relevancy: Optional[float] = None
    answer_correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_relevancy: Optional[float] = None
    answer_similarity: Optional[float] = None
    context_entity_recall: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }
    
    def get_overall_score(self) -> float:
        """Calculate weighted overall score"""
        scores = []
        weights = []
        
        # Define metric weights
        metric_weights = {
            'answer_relevancy': 0.25,
            'faithfulness': 0.25,
            'context_precision': 0.20,
            'context_recall': 0.15,
            'answer_correctness': 0.15
        }
        
        for metric, weight in metric_weights.items():
            value = getattr(self, metric)
            if value is not None:
                scores.append(value)
                weights.append(weight)
        
        if not scores:
            return 0.0
        
        # Weighted average
        return np.average(scores, weights=weights)

@dataclass
class EvaluationResult:
    """Result of RAGAS evaluation"""
    query: str
    generated_answer: str
    reference_answer: Optional[str]
    retrieved_contexts: List[str]
    metrics: RAGASMetrics
    individual_scores: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchEvaluationResult:
    """Result of batch RAGAS evaluation"""
    results: List[EvaluationResult]
    aggregate_metrics: RAGASMetrics
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class RAGASEvaluator:
    """Main RAGAS evaluation orchestrator"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not available. Please install with: pip install ragas")
        
        self.model_name = model_name
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> List[Any]:
        """Initialize RAGAS metrics"""
        
        return [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
            context_relevancy,
            answer_similarity,
            context_entity_recall
        ]
    
    async def evaluate_single(
        self,
        query: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        reference_answer: Optional[str] = None,
        ground_truths: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Evaluate a single RAG response"""
        
        logger.debug(f"Evaluating single query: {query[:50]}...")
        
        # Prepare dataset
        data = {
            'question': [query],
            'answer': [generated_answer],
            'contexts': [retrieved_contexts],
        }
        
        if ground_truths:
            data['ground_truths'] = [ground_truths]
        elif reference_answer:
            data['ground_truths'] = [[reference_answer]]
        else:
            # Use generated answer as ground truth for metrics that require it
            data['ground_truths'] = [[generated_answer]]
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        try:
            start_time = datetime.now()
            
            # Select appropriate metrics based on available data
            metrics_to_use = self._select_metrics(reference_answer is not None or ground_truths is not None)
            
            evaluation = evaluate(dataset, metrics=metrics_to_use)
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            # Extract metrics
            metrics = self._extract_metrics(evaluation)
            
            return EvaluationResult(
                query=query,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts,
                metrics=metrics,
                individual_scores=evaluation,
                metadata={
                    'evaluation_time': evaluation_time,
                    'num_contexts': len(retrieved_contexts),
                    'answer_length': len(generated_answer)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            # Return empty metrics on error
            return EvaluationResult(
                query=query,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts,
                metrics=RAGASMetrics(),
                metadata={'error': str(e)}
            )
    
    async def evaluate_batch(
        self,
        queries: List[str],
        generated_answers: List[str],
        retrieved_contexts_list: List[List[str]],
        reference_answers: Optional[List[str]] = None,
        ground_truths_list: Optional[List[List[str]]] = None
    ) -> BatchEvaluationResult:
        """Evaluate a batch of RAG responses"""
        
        logger.info(f"Evaluating batch of {len(queries)} queries")
        
        start_time = datetime.now()
        
        # Prepare dataset
        data = {
            'question': queries,
            'answer': generated_answers,
            'contexts': retrieved_contexts_list,
        }
        
        # Add ground truths if available
        if ground_truths_list:
            data['ground_truths'] = ground_truths_list
        elif reference_answers:
            data['ground_truths'] = [[ref] for ref in reference_answers]
        else:
            # Use generated answers as ground truths for metrics that require them
            data['ground_truths'] = [[ans] for ans in generated_answers]
        
        dataset = Dataset.from_dict(data)
        
        try:
            # Select appropriate metrics
            has_reference = reference_answers is not None or ground_truths_list is not None
            metrics_to_use = self._select_metrics(has_reference)
            
            # Run batch evaluation
            evaluation = evaluate(dataset, metrics=metrics_to_use)
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            # Create individual results
            individual_results = []
            for i in range(len(queries)):
                # Extract metrics for this item
                item_metrics = self._extract_item_metrics(evaluation, i)
                
                result = EvaluationResult(
                    query=queries[i],
                    generated_answer=generated_answers[i],
                    reference_answer=reference_answers[i] if reference_answers else None,
                    retrieved_contexts=retrieved_contexts_list[i],
                    metrics=item_metrics,
                    individual_scores={},
                    metadata={
                        'index': i,
                        'num_contexts': len(retrieved_contexts_list[i])
                    }
                )
                individual_results.append(result)
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(individual_results)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(individual_results, evaluation_time)
            
            return BatchEvaluationResult(
                results=individual_results,
                aggregate_metrics=aggregate_metrics,
                summary_stats=summary_stats,
                evaluation_time=evaluation_time,
                metadata={
                    'batch_size': len(queries),
                    'metrics_used': [m.name for m in metrics_to_use],
                    'evaluation_model': self.model_name
                }
            )
            
        except Exception as e:
            logger.error(f"Error in batch RAGAS evaluation: {e}")
            # Return empty results on error
            return BatchEvaluationResult(
                results=[],
                aggregate_metrics=RAGASMetrics(),
                summary_stats={'error': str(e)},
                evaluation_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    def _select_metrics(self, has_reference: bool) -> List[Any]:
        """Select appropriate metrics based on available data"""
        
        # Always available metrics (don't require ground truth)
        base_metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_relevancy
        ]
        
        # Metrics that require ground truth/reference
        reference_metrics = [
            answer_correctness,
            context_recall,
            answer_similarity,
            context_entity_recall
        ]
        
        if has_reference:
            return base_metrics + reference_metrics
        else:
            return base_metrics
    
    def _extract_metrics(self, evaluation: Dict[str, Any]) -> RAGASMetrics:
        """Extract metrics from RAGAS evaluation result"""
        
        return RAGASMetrics(
            answer_relevancy=evaluation.get('answer_relevancy'),
            answer_correctness=evaluation.get('answer_correctness'),
            faithfulness=evaluation.get('faithfulness'),
            context_precision=evaluation.get('context_precision'),
            context_recall=evaluation.get('context_recall'),
            context_relevancy=evaluation.get('context_relevancy'),
            answer_similarity=evaluation.get('answer_similarity'),
            context_entity_recall=evaluation.get('context_entity_recall')
        )
    
    def _extract_item_metrics(self, evaluation: Dict[str, Any], index: int) -> RAGASMetrics:
        """Extract metrics for a specific item from batch evaluation"""
        
        # For batch evaluations, metrics are arrays
        metrics = RAGASMetrics()
        
        for metric_name in ['answer_relevancy', 'answer_correctness', 'faithfulness',
                           'context_precision', 'context_recall', 'context_relevancy',
                           'answer_similarity', 'context_entity_recall']:
            if metric_name in evaluation:
                values = evaluation[metric_name]
                if isinstance(values, (list, np.ndarray)) and len(values) > index:
                    setattr(metrics, metric_name, float(values[index]))
                elif isinstance(values, (int, float)):
                    setattr(metrics, metric_name, float(values))
        
        return metrics
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> RAGASMetrics:
        """Calculate aggregate metrics across all results"""
        
        if not results:
            return RAGASMetrics()
        
        # Collect all non-null values for each metric
        metric_values = {}
        
        for result in results:
            for metric_name in ['answer_relevancy', 'answer_correctness', 'faithfulness',
                               'context_precision', 'context_recall', 'context_relevancy',
                               'answer_similarity', 'context_entity_recall']:
                value = getattr(result.metrics, metric_name)
                if value is not None:
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
        
        # Calculate means
        aggregate = RAGASMetrics()
        for metric_name, values in metric_values.items():
            if values:
                setattr(aggregate, metric_name, float(np.mean(values)))
        
        return aggregate
    
    def _calculate_summary_stats(
        self,
        results: List[EvaluationResult],
        evaluation_time: float
    ) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        if not results:
            return {}
        
        # Overall scores
        overall_scores = [r.metrics.get_overall_score() for r in results]
        
        # Context statistics
        context_counts = [len(r.retrieved_contexts) for r in results]
        answer_lengths = [len(r.generated_answer) for r in results]
        
        return {
            'num_queries': len(results),
            'evaluation_time_seconds': evaluation_time,
            'avg_evaluation_time_per_query': evaluation_time / len(results),
            'overall_score_mean': float(np.mean(overall_scores)),
            'overall_score_std': float(np.std(overall_scores)),
            'overall_score_min': float(np.min(overall_scores)),
            'overall_score_max': float(np.max(overall_scores)),
            'avg_contexts_per_query': float(np.mean(context_counts)),
            'avg_answer_length': float(np.mean(answer_lengths)),
            'queries_with_errors': sum(1 for r in results if 'error' in r.metadata)
        }

class RAGASReporter:
    """Generate reports from RAGAS evaluation results"""
    
    @staticmethod
    def generate_detailed_report(result: BatchEvaluationResult) -> str:
        """Generate a detailed text report"""
        
        report = []
        report.append("RAGAS Evaluation Report")
        report.append("=" * 50)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Queries: {len(result.results)}")
        report.append(f"Evaluation Time: {result.evaluation_time:.2f} seconds")
        report.append("")
        
        # Aggregate metrics
        report.append("Aggregate Metrics:")
        report.append("-" * 20)
        aggregate_dict = result.aggregate_metrics.to_dict()
        for metric, value in aggregate_dict.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        overall_score = result.aggregate_metrics.get_overall_score()
        report.append(f"Overall Score: {overall_score:.3f}")
        report.append("")
        
        # Summary statistics
        if result.summary_stats:
            report.append("Summary Statistics:")
            report.append("-" * 20)
            for stat, value in result.summary_stats.items():
                if isinstance(value, float):
                    report.append(f"{stat.replace('_', ' ').title()}: {value:.3f}")
                else:
                    report.append(f"{stat.replace('_', ' ').title()}: {value}")
            report.append("")
        
        # Individual results (top 5 and bottom 5)
        if result.results:
            sorted_results = sorted(result.results, key=lambda x: x.metrics.get_overall_score(), reverse=True)
            
            report.append("Top 5 Performing Queries:")
            report.append("-" * 30)
            for i, res in enumerate(sorted_results[:5]):
                report.append(f"{i+1}. Query: {res.query[:80]}...")
                report.append(f"   Overall Score: {res.metrics.get_overall_score():.3f}")
                report.append("")
            
            if len(sorted_results) > 5:
                report.append("Bottom 5 Performing Queries:")
                report.append("-" * 30)
                for i, res in enumerate(sorted_results[-5:]):
                    report.append(f"{i+1}. Query: {res.query[:80]}...")
                    report.append(f"   Overall Score: {res.metrics.get_overall_score():.3f}")
                    report.append("")
        
        return "\n".join(report)
    
    @staticmethod
    def save_results_to_json(result: BatchEvaluationResult, file_path: str):
        """Save results to JSON file"""
        
        # Convert to serializable format
        data = {
            'metadata': result.metadata,
            'evaluation_time': result.evaluation_time,
            'aggregate_metrics': result.aggregate_metrics.to_dict(),
            'summary_stats': result.summary_stats,
            'results': []
        }
        
        for res in result.results:
            data['results'].append({
                'query': res.query,
                'generated_answer': res.generated_answer,
                'reference_answer': res.reference_answer,
                'retrieved_contexts': res.retrieved_contexts,
                'metrics': res.metrics.to_dict(),
                'metadata': res.metadata
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"RAGAS results saved to {file_path}")
    
    @staticmethod
    def create_dataframe(result: BatchEvaluationResult) -> pd.DataFrame:
        """Create pandas DataFrame from results"""
        
        if not result.results:
            return pd.DataFrame()
        
        data = []
        for res in result.results:
            row = {
                'query': res.query,
                'answer_length': len(res.generated_answer),
                'num_contexts': len(res.retrieved_contexts),
                'overall_score': res.metrics.get_overall_score()
            }
            
            # Add individual metrics
            row.update(res.metrics.to_dict())
            
            # Add metadata
            row.update({f"meta_{k}": v for k, v in res.metadata.items()})
            
            data.append(row)
        
        return pd.DataFrame(data)

# Global instance
ragas_evaluator = None

def get_ragas_evaluator(model_name: str = "gpt-3.5-turbo") -> RAGASEvaluator:
    """Get RAGAS evaluator instance"""
    global ragas_evaluator
    
    if ragas_evaluator is None:
        ragas_evaluator = RAGASEvaluator(model_name)
    
    return ragas_evaluator