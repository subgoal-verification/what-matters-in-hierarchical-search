# What Matters in Hierarchical Search for Combinatorial Reasoning Problems?

In this study, we extensively test which properties of the environments and training datasets make them amenable to subgoal methods. Furthermore, we verify the common beliefs present in the literature regarding that class of algorithms.
We identify the attributes pivotal for leveraging the advantages of high-level search: *presence of dead ends in the environment, hard-to-learn value functions, complex action spaces, or using data collected from diverse experts*. As an additional outcome of our experiments, we discover several pitfalls in evaluation that are easy to overlook but lead to misleading conclusions. To solve this problem, we propose a consistent evaluation methodology, which allows for a meaningful comparison between various methods.

Our main contributions are the following:
- We identify the key properties of environments and datasets that impact the performance of hierarchical search methods.
- We analyze the suitability of HRL algorithms for the domain of combinatorial reasoning and list the necessary attributes.
- We highlight the critical points for reliable reporting of the results in this domain.

This repository contains code essential for reproducing results presented in our paper. 

## Running code

To run the code locally, simply run the command below

`python3 -m carl.run --config_file config_file_name.yaml`

## Example config

Below there is an example of BestFS algorithm config.

```
algorithm:
  _target_: carl.solve_instances.solve_instances.SolveInstances

  # solver class
  solver_class:
    _target_: carl.solver.subgoal_search.Solver
    max_nodes: 2000
    planner_class:
      _partial_: true
      _target_: carl.solver.planners.BestFSPlanner

    subgoal_generator:
      _target_: carl.inference_components.policy.PolicyGeneratorWrapper
      policy:
        _target_: carl.inference_components.policy.TransformerPolicy
        policy_network:
          _target_: transformers.BertForSequenceClassification
          config:
            _target_: transformers.BertConfig
        path_to_policy_weights: path_to_policy_weights
        env:
          _target_: carl.environment.sokoban.env.SokobanEnv
          tokenizer:
            _target_: carl.environment.sokoban.tokenizer.SokobanTokenizer
            cut_distance: 150
            type_of_value_training: "regression"
            size_of_board:
              - 12
              - 12
          num_boxes: 4
        # if confidence_threshold is None, then the n_actions parameter is used.
        n_actions: None
        confidence_threshold: 0.0
      env:
        _target_: carl.environment.sokoban.env.SokobanEnv
        tokenizer:
          _target_: carl.environment.sokoban.tokenizer.SokobanTokenizer
          cut_distance: 150
          type_of_value_training: "regression"
          size_of_board:
            - 12
            - 12
        num_boxes: 4

    # validator
    validator:
      _target_: carl.inference_components.validator.DummyValidator
      env:
        _target_: carl.environment.sokoban.env.SokobanEnv
        tokenizer:
          _target_: carl.environment.sokoban.tokenizer.SokobanTokenizer
          cut_distance: 150
          type_of_value_training: "regression"
          size_of_board:
            - 12
            - 12
        num_boxes: 4

    # value function
    value_function:
      _target_: carl.inference_components.value.TransformerValue
      value_network:
        _target_: transformers.BertForSequenceClassification
        config:
          _target_: transformers.BertConfig
      path_to_value_network_weights: path_to_value_weights
      type_of_evaluation: "regression"
      noise_variance: 0.85
      env:
        _target_: carl.environment.sokoban.env.SokobanEnv
        tokenizer:
          _target_: carl.environment.sokoban.tokenizer.SokobanTokenizer
          cut_distance: 150
          type_of_value_training: "regression"
          size_of_board:
            - 12
            - 12
        num_boxes: 4

  # data loader class
  data_loader_class:
    _target_: carl.environment.instance_generator.BasicInstanceGenerator
    generator:
      _target_: carl.environment.instance_generator.GeneralIterableDataLoader
      path_to_folder_with_data: path_to_folder_with_data
    batch_size: 1

  # result logger
  result_logger: null

  n_parallel_workers: 1

```
## Training components

To train components use following configs (example for sokoban environment):
- value function `sokoban_train_value.yaml`
- subgoal generator `sokoban_train_generator.yaml`
- conditional low-level policy `sokoban_train_cllp.yaml`
- behavioral cloning policy `sokoban_train_policy.yaml`
