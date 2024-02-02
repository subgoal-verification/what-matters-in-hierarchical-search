from typing import Any
import joblib
from loguru import logger

from carl.environment.instance_generator import BasicInstanceGenerator
from carl.solve_instances.result_loggers import ResultLogger
from carl.solver.subgoal_search import Solver
from carl.solver.subgoal_search_batched import SubgoalSearchSolver


class SolveInstances:
    def __init__(
        self,
        solver: SubgoalSearchSolver,
        data_loader: BasicInstanceGenerator,
        result_logger: ResultLogger,
        problems_to_solve: int,
        n_jobs: int,
        dump_solved=False,
    ):
        self.solver = solver
        self.data_loader = data_loader
        self.result_logger = result_logger

        self.problems_to_solve = problems_to_solve

        self.completed_problems: int = 0
        self.dump_solved = dump_solved
        self.n_jobs = n_jobs

    def run(self) -> None:
        logger.warning('Running solve_instances.py')
        
        self.solver.construct_networks()
        
        initial_state_loader = iter(self.data_loader.reset_dataloader())
        inputs = []
        for _ in range(self.problems_to_solve):
            initial_state = next(initial_state_loader).cpu().numpy()[0]
            print(initial_state.shape)
            inputs.append(initial_state)
        if isinstance(self.solver, SubgoalSearchSolver):
            neptune_callback = self.result_logger.custom_logger
            results = self.solver.solve(inputs, neptune_callback)
        else:
            assert isinstance(self.solver, Solver)
            neptune_callback = self.result_logger.custom_logger
            for i, result in enumerate(joblib.Parallel(n_jobs=self.n_jobs, verbose=100, return_as='generator')(
                joblib.delayed(self.solver.solve)(initial_state)
                for initial_state in inputs
            )):
                self.result_logger.log_results([result])
                if self.dump_solved:
                    joblib.dump(result, f'solved_problems_{i}.joblib')
