# Progressive State Space Disaggregation for Infinite Horizon Dynamic Programming

How to use the repository in three steps :
- pip install -r requirements.txt
- python main.py
- observe results in results/discounted_avg_runtime_ICAPS_agg_0.xlsx

Models implemented here :
- Tandem Queue (Tournaire, Thomas, et al. "Factored reinforcement learning for auto-scaling in tandem queues." NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium. IEEE, 2022.)
- Four Rooms (Hengst, Bernhard. "Hierarchical approaches." Reinforcement Learning: State-of-the-Art. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012. 293-323.)
- Randomly generated MDPs (Archibald, T. W., K. I. M. McKinnon, and L. C. Thomas. "On the generation of markov decision processes." Journal of the Operational Research Society 46.3 (1995): 354-361.)
- Sutton tracks (Sutton, Barto, Reinforcement Learning : an Introduction Example 5.4 p.127)
- Barto tracks (Barto, Andrew G., Steven J. Bradtke, and Satinder P. Singh. "Learning to act using real-time dynamic programming." Artificial intelligence 72.1-2 (1995): 81-138.)
- Queue with Impatience (Jean-Marie, A., & Hyon, E. (2009). Scheduling in a queuing system with impatience and setup costs)
- Garnet MDPs (Piot, Bilal, Matthieu Geist, and Olivier Pietquin. "Bridging the gap between imitation learning and inverse reinforcement learning." IEEE transactions on neural networks and learning systems 28.8 (2016): 1814-1826.)

Remarks :
- Whole experimence takes several days as runtime is measured on 10 experience for each model and each solver
- discount = 0.999, epsilon = 1e-3
- All solvers are implemented in the "solvers" folder, and all models in the "models" solver