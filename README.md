# DCBO

This repository is the official implementation of the paper ["Dependence in constrained Bayesian optimization: When do we need it and how does it help?"](https://link.springer.com/article/10.1007/s11590-023-02047-z).

The BibTex reference is:

    @article{zhang2023dependence,
        title={Dependence in constrained Bayesian optimization: When do we need it and how does it help?},
        author={Zhang, Shiqiang and Lee, Robert M and Shafei, Behrang and Walz, David and Misener, Ruth},
        journal={Optimization Letters},
        pages={1--17},
        year={2023},
        publisher={Springer}
    }


To install requirements:

```
pip install -r requirements.txt
```

To optimize a function with different methods shown in the paper, run this command:

```
python main.py $fun_index $method_index $budget
```
where \$fun_index is the index of function, \$method_index is the index of method, and \$budget is the number of iterations.


$16$ benchmarks and $6$ methods are supported. One can check these benchmarks and methods in `main.py`.


The rest of the files correspond to:

 - acquisitions.py: implements six acquisitions used in the paper, consist of two unconstrained acquisitions (`constrained_expected_improvement` and `constrained_adaptive_sampling`) and three ways to calculate the possibility of feasibility (`independent_probability_of_feasibility`, `dependent_probability_of_feasibility`, and `independent_probability_of_feasibility_MOGP`).

 - functions.py: defines all benchmarks used in the paper.

 - models.py: implements two models used in the paper (`Independent_MOGP` and `Dependent_MOGP`). The first one consists of multiple Gaussian processes, the second one is a multiple output Gaussian process.

 - plot_utils.py: plots numerical results.

# Contributors
Shiqiang Zhang. Funded by an Imperial College Hans Rausing PhD Scholarship.
