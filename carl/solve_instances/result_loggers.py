import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Any

from carl.custom_logger.loggers import NeptuneCaRLLogger
from carl.solve_instances.metric_logging import MetricsAccumulator


class ResultLogger(ABC):
    @abstractmethod
    def log_results(self, results: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class SubgoalSearchResultLogger(ResultLogger):
    def __init__(self, custom_logger: NeptuneCaRLLogger, budget_logs: list[int]) -> None:
        self.custom_logger = custom_logger.return_logger()
        self.completed_problems: int = 0
        self.solved_problems: int = 0
        self.finished_reasons: dict[str, int] = {}
        self.solve_budgets: list[int] = []
        self.budget_logs = budget_logs
        self.solved_stats: MetricsAccumulator = MetricsAccumulator()

    def log_results(self, results) -> None:
        for task_id, (solution, search_info) in enumerate(results):
            # Log the main solved rate metric and the solution.
            self.solved_stats.log_metric_to_average('rate/full', solution['solved'])
            self.solved_problems += solution['solved']
            self.custom_logger.run['binary_solved'].append(1 if solution['solved'] else 0)
            
            # Log subgoal generation number
            for k, v in search_info['subgoals_added'].items():
                self.custom_logger.run[f'subgoals_added_to_search_tree/{k}'].append(v)
                    
            # Log the number of subgoals selected for expansion.
            for k, v in search_info['subgoals_selected_for_expansion'].items():
                self.custom_logger.run[f'subgoals_selected_for_expansion/{k}'].append(value=v)


            # Log solved rates for selected budgets.
            for budget in self.budget_logs:
                is_solved_in_budget = solution['solved'] and (
                    search_info['nodes_visited'] <= budget
                )
                self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', is_solved_in_budget)
                
                # High Level budget
                is_solved_in_high_level_budget = solution['solved'] and (
                    search_info['subgoals_visited'] <= budget
                )
                self.solved_stats.log_metric_to_average(f'rate/{budget}_subgoals', is_solved_in_high_level_budget)

            # Log additional search metrics.
            for metric, value in search_info.items():
                if isinstance(value, int | float):
                    self.custom_logger.run[metric].append(value)

            # Log fraction of reachable nodes and unreachable nodes.
            fraction_of_reachable_nodes: float = search_info['nodes_valid'] / (
                search_info['nodes_valid'] + search_info['nodes_unreachable']
            )
            fraction_of_unreachable_nodes: float = search_info['nodes_unreachable'] / (
                search_info['nodes_valid'] + search_info['nodes_unreachable']
            )

            self.custom_logger.run['fraction_of_reachable_nodes'].append(fraction_of_reachable_nodes)
            self.custom_logger.run['fraction_of_unreachable_nodes'].append(fraction_of_unreachable_nodes)

            # Count the finished reasons.
            finished_reason = search_info['finished_reason']
            self.finished_reasons[finished_reason] = (
                self.finished_reasons.get(finished_reason, 0) + 1
            )

        self.completed_problems = self.completed_problems + len(results)
        self.custom_logger.run['total_completed_problems'].append(self.completed_problems)
        # Log the solved rate metrics.
        for metric, value in self.solved_stats.return_scalars().items():
            self.custom_logger.run[join('solved', metric)].append(value)
        
        self.custom_logger.run['problems/solved'].append(self.solved_problems)

        # Log the finished reasons.
        for finished_reason, count in self.finished_reasons.items():
            self.custom_logger.run[f'finished_reasons/{finished_reason}/rate'
            ].append(count / self.completed_problems)
    
    
    def reset(self) -> None:
        # reset the metrics
        self.solved_stats = MetricsAccumulator()
